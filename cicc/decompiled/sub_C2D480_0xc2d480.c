// Function: sub_C2D480
// Address: 0xc2d480
//
__int64 __fastcall sub_C2D480(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r15
  unsigned int v5; // eax
  unsigned int v6; // ebx
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 *v10; // rax
  __int64 *v11; // r12
  __int64 v12; // rsi
  __int64 *v13; // rbx
  unsigned __int64 v14; // rcx
  __int64 *v15; // rax
  __int64 v16; // rax
  int v17; // esi
  __int64 v18; // rax
  unsigned int v19; // r11d
  __int64 *v20; // rdx
  __int64 v21; // rcx
  int v22; // r8d
  __int64 *v23; // r10
  int v24; // ecx
  __int64 v25; // [rsp+10h] [rbp-110h]
  __int64 v26[2]; // [rsp+20h] [rbp-100h] BYREF
  __int64 v27; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v28; // [rsp+38h] [rbp-E8h]
  __int64 v29; // [rsp+40h] [rbp-E0h]
  __int64 v30; // [rsp+48h] [rbp-D8h]
  _QWORD v31[26]; // [rsp+50h] [rbp-D0h] BYREF

  v3 = a1;
  *(_QWORD *)(a1 + 208) = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 216) = *(_QWORD *)(a1 + 168);
  v5 = sub_C2BD40(a1, a2, a3);
  if ( v5 )
    return v5;
  v8 = *(_QWORD *)(a1 + 208);
  v27 = 0;
  v28 = 0;
  *(_QWORD *)(a1 + 216) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 16);
  v29 = 0;
  v30 = 0;
  if ( !(_DWORD)v8 || (v10 = *(__int64 **)(a2 + 8), v11 = &v10[2 * *(unsigned int *)(a2 + 24)], v10 == v11) )
  {
LABEL_5:
    v9 = sub_C27C10((_QWORD *)v3, *(_BYTE *)(v3 + 176), (__int64)&v27);
  }
  else
  {
    while ( 1 )
    {
      v12 = *v10;
      v13 = v10;
      if ( *v10 != -1 && v12 != -2 )
        break;
      v10 += 2;
      if ( v11 == v10 )
        goto LABEL_13;
    }
    if ( v11 != v10 )
    {
      while ( 1 )
      {
        v14 = v13[1];
        if ( v12 )
        {
          v25 = v13[1];
          sub_C7D030(v31);
          sub_C7D280(v31, v12, v25);
          sub_C7D290(v31, v26);
          v14 = v26[0];
        }
        v31[0] = v14;
        v15 = sub_C1DD00(a3, v14 % a3[1], v31, v14);
        if ( v15 )
        {
          v16 = *v15;
          if ( v16 )
            break;
        }
LABEL_21:
        v13 += 2;
        if ( v13 != v11 )
        {
          while ( 1 )
          {
            v12 = *v13;
            if ( *v13 != -1 && v12 != -2 )
              break;
            v13 += 2;
            if ( v11 == v13 )
              goto LABEL_25;
          }
          if ( v11 != v13 )
            continue;
        }
LABEL_25:
        v3 = a1;
        goto LABEL_5;
      }
      v17 = v30;
      v18 = v16 + 16;
      v26[0] = v18;
      if ( (_DWORD)v30 )
      {
        v19 = (v30 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
        v20 = (__int64 *)(v28 + 8LL * v19);
        v21 = *v20;
        if ( v18 == *v20 )
          goto LABEL_21;
        v22 = 1;
        v23 = 0;
        while ( v21 != -4096 )
        {
          if ( v23 || v21 != -8192 )
            v20 = v23;
          v19 = (v30 - 1) & (v22 + v19);
          v21 = *(_QWORD *)(v28 + 8LL * v19);
          if ( v18 == v21 )
            goto LABEL_21;
          ++v22;
          v23 = v20;
          v20 = (__int64 *)(v28 + 8LL * v19);
        }
        if ( !v23 )
          v23 = v20;
        ++v27;
        v24 = v29 + 1;
        v31[0] = v23;
        if ( 4 * ((int)v29 + 1) < (unsigned int)(3 * v30) )
        {
          if ( (int)v30 - HIDWORD(v29) - v24 > (unsigned int)v30 >> 3 )
          {
LABEL_34:
            LODWORD(v29) = v24;
            if ( *v23 != -4096 )
              --HIDWORD(v29);
            *v23 = v18;
            goto LABEL_21;
          }
LABEL_39:
          sub_C2D2B0((__int64)&v27, v17);
          sub_C260C0((__int64)&v27, v26, v31);
          v18 = v26[0];
          v23 = (__int64 *)v31[0];
          v24 = v29 + 1;
          goto LABEL_34;
        }
      }
      else
      {
        ++v27;
        v31[0] = 0;
      }
      v17 = 2 * v30;
      goto LABEL_39;
    }
LABEL_13:
    v9 = sub_C27C10((_QWORD *)a1, *(_BYTE *)(a1 + 176), (__int64)&v27);
  }
  v6 = v9;
  if ( !v9 )
  {
    v6 = 0;
    sub_C1AFD0();
  }
  sub_C7D6A0(v28, 8LL * (unsigned int)v30, 8);
  return v6;
}
