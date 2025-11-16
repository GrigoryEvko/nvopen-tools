// Function: sub_259E860
// Address: 0x259e860
//
__int64 __fastcall sub_259E860(__int64 a1, __int64 a2)
{
  char *v3; // r12
  char v4; // al
  __int64 v5; // rsi
  unsigned __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rbx
  _QWORD *v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r8
  int v15; // esi
  int v16; // r15d
  __int64 v17; // r9
  __int64 *v18; // r11
  unsigned int v19; // edx
  __int64 *v20; // rdi
  __int64 v21; // rcx
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // r15
  unsigned __int64 v25; // rdx
  __int64 v27; // [rsp+18h] [rbp-B8h]
  __int64 v28; // [rsp+20h] [rbp-B0h] BYREF
  __int64 *v29; // [rsp+28h] [rbp-A8h] BYREF
  __int64 v30; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-98h]
  __int64 v32; // [rsp+40h] [rbp-90h]
  __int64 v33; // [rsp+48h] [rbp-88h]
  _QWORD *v34; // [rsp+50h] [rbp-80h] BYREF
  __int64 v35; // [rsp+58h] [rbp-78h]
  _BYTE v36[112]; // [rsp+60h] [rbp-70h] BYREF

  v3 = (char *)sub_250D070((_QWORD *)(a1 + 72));
  v4 = *v3;
  if ( (unsigned __int8)*v3 <= 0x1Cu )
    return 1;
  if ( v4 != 62 )
  {
    if ( v4 == 64 || sub_259E650(a1, a2, (unsigned __int64)v3) && *v3 != 34 )
    {
      sub_2570110(a2, (__int64)v3);
      return 0;
    }
    return 1;
  }
  v34 = v36;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v35 = 0x800000000LL;
  sub_2574220(a1, a2, (__int64)v3, (__int64)&v30);
  v5 = (__int64)v3;
  v6 = 0;
  sub_2570110(a2, v5);
  if ( (_DWORD)v35 )
  {
    v27 = a2;
    while ( 1 )
    {
      v7 = v34[v6];
      v8 = *(_QWORD *)(v7 + 16);
      if ( v8 )
        break;
LABEL_23:
      ++v6;
      sub_2570110(v27, v7);
      if ( (unsigned int)v35 <= v6 )
        goto LABEL_24;
    }
    while ( 1 )
    {
      v14 = *(_QWORD *)(v8 + 24);
      v28 = v14;
      if ( !(_DWORD)v32 )
        break;
      v15 = v33;
      if ( !(_DWORD)v33 )
      {
        ++v30;
        v29 = 0;
LABEL_30:
        v15 = 2 * v33;
LABEL_31:
        sub_CF4090((__int64)&v30, v15);
        sub_23FDF60((__int64)&v30, &v28, &v29);
        v14 = v28;
        v18 = v29;
        v22 = v32 + 1;
        goto LABEL_18;
      }
      v16 = 1;
      v17 = v31;
      v18 = 0;
      v19 = (v33 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v20 = (__int64 *)(v31 + 8LL * v19);
      v21 = *v20;
      if ( v14 == *v20 )
      {
LABEL_9:
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          goto LABEL_23;
      }
      else
      {
        while ( v21 != -4096 )
        {
          if ( v18 || v21 != -8192 )
            v20 = v18;
          v19 = (v33 - 1) & (v16 + v19);
          v21 = *(_QWORD *)(v31 + 8LL * v19);
          if ( v14 == v21 )
            goto LABEL_9;
          ++v16;
          v18 = v20;
          v20 = (__int64 *)(v31 + 8LL * v19);
        }
        if ( !v18 )
          v18 = v20;
        v22 = v32 + 1;
        ++v30;
        v29 = v18;
        if ( 4 * ((int)v32 + 1) >= (unsigned int)(3 * v33) )
          goto LABEL_30;
        if ( (int)v33 - HIDWORD(v32) - v22 <= (unsigned int)v33 >> 3 )
          goto LABEL_31;
LABEL_18:
        LODWORD(v32) = v22;
        if ( *v18 != -4096 )
          --HIDWORD(v32);
        *v18 = v14;
        v23 = (unsigned int)v35;
        v24 = v28;
        v25 = (unsigned int)v35 + 1LL;
        if ( v25 > HIDWORD(v35) )
        {
          sub_C8D5F0((__int64)&v34, v36, v25, 8u, v14, v17);
          v23 = (unsigned int)v35;
        }
        v34[v23] = v24;
        LODWORD(v35) = v35 + 1;
        v8 = *(_QWORD *)(v8 + 8);
        if ( !v8 )
          goto LABEL_23;
      }
    }
    v9 = &v34[(unsigned int)v35];
    if ( v9 == sub_2537FC0(v34, (__int64)v9, &v28) )
      sub_2573C90((__int64)&v30, v12, v10, v11, v12, v13);
    goto LABEL_9;
  }
LABEL_24:
  if ( v34 != (_QWORD *)v36 )
    _libc_free((unsigned __int64)v34);
  sub_C7D6A0(v31, 8LL * (unsigned int)v33, 8);
  return 0;
}
