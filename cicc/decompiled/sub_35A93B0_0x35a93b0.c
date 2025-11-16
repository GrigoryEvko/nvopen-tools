// Function: sub_35A93B0
// Address: 0x35a93b0
//
__int64 __fastcall sub_35A93B0(__int64 *a1)
{
  __int64 *v1; // r15
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  __int64 *v4; // rdi
  __int64 v5; // r13
  int v6; // eax
  _BYTE *v7; // r12
  _BYTE *v8; // r14
  _BYTE *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 *v14; // rdi
  unsigned int v15; // ebx
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rax
  unsigned int v19; // r11d
  __int64 *v20; // rdx
  __int64 v21; // r14
  int v22; // r14d
  unsigned int v23; // r14d
  bool v24; // r15
  unsigned int v25; // r14d
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rcx
  __int64 v29; // rdx
  _BYTE *v30; // r12
  int v31; // edx
  int v32; // ecx
  __int64 *v34; // [rsp+0h] [rbp-80h]
  __int64 *v35; // [rsp+8h] [rbp-78h]
  __int64 *v36; // [rsp+18h] [rbp-68h]
  _BYTE *v37; // [rsp+20h] [rbp-60h]
  _BYTE *v38; // [rsp+28h] [rbp-58h]
  bool v39; // [rsp+33h] [rbp-4Dh]
  int v40; // [rsp+34h] [rbp-4Ch]
  unsigned int v41; // [rsp+44h] [rbp-3Ch] BYREF
  unsigned int *v42; // [rsp+48h] [rbp-38h] BYREF

  v1 = a1;
  v2 = sub_2EA6400(*(_QWORD *)*a1);
  a1[6] = (__int64)v2;
  v3 = *(_QWORD **)v2[8];
  a1[7] = (__int64)v3;
  if ( v2 == v3 )
    a1[7] = *(_QWORD *)(v2[8] + 8LL);
  v4 = (__int64 *)*a1;
  v34 = *(__int64 **)(*v1 + 16);
  if ( v34 != *(__int64 **)(*v1 + 8) )
  {
    v35 = *(__int64 **)(*v1 + 8);
    v36 = v1 + 11;
    while ( 1 )
    {
      v5 = *v35;
      v6 = sub_3598DB0((__int64)v4, *v35);
      v7 = *(_BYTE **)(v5 + 32);
      v40 = v6;
      v8 = &v7[40 * (*(_DWORD *)(v5 + 40) & 0xFFFFFF)];
      v37 = v8;
      if ( v7 != v8 )
      {
        while ( 1 )
        {
          v9 = v7;
          if ( sub_2DADC00(v7) )
            break;
          v7 += 40;
          if ( v8 == v7 )
            goto LABEL_42;
        }
        if ( v8 != v7 )
          break;
      }
LABEL_42:
      if ( v34 == ++v35 )
        return sub_35A8C50(v1);
      v4 = (__int64 *)*v1;
    }
    while ( 1 )
    {
      v10 = *((unsigned int *)v9 + 2);
      v11 = v1[3];
      v41 = v10;
      if ( (int)v10 < 0 )
        v12 = *(_QWORD *)(*(_QWORD *)(v11 + 56) + 16 * (v10 & 0x7FFFFFFF) + 8);
      else
        v12 = *(_QWORD *)(*(_QWORD *)(v11 + 304) + 8 * v10);
      if ( !v12 )
        goto LABEL_46;
      if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(v12 + 32);
          if ( !v12 )
            break;
          if ( (*(_BYTE *)(v12 + 3) & 0x10) == 0 )
            goto LABEL_14;
        }
LABEL_46:
        v26 = v1[12];
        v39 = 0;
        v25 = 0;
        if ( v26 )
          goto LABEL_30;
        goto LABEL_47;
      }
LABEL_14:
      v39 = 0;
      v13 = v12;
      v14 = v1;
      v38 = v9;
      v15 = 0;
LABEL_15:
      v16 = *(_QWORD *)(v13 + 16);
      v17 = *(_QWORD *)(*v14 + 72);
      v18 = *(unsigned int *)(*v14 + 88);
      if ( !(_DWORD)v18 )
        goto LABEL_50;
      v19 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v16 != *v20 )
        break;
LABEL_17:
      if ( v20 == (__int64 *)(v17 + 16 * v18) )
        goto LABEL_50;
      v22 = *((_DWORD *)v20 + 2);
      if ( v22 == -1 || v40 > v22 )
        goto LABEL_50;
      v23 = v22 - v40;
LABEL_21:
      v24 = *(_WORD *)(v5 + 68) == 68 || *(_WORD *)(v5 + 68) == 0;
      if ( v24 )
      {
        if ( (unsigned __int8)sub_3599670(v14, v5) )
          ++v23;
        else
          v39 = v24;
      }
      if ( v15 < v23 )
        v15 = v23;
      while ( 1 )
      {
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          break;
        if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
          goto LABEL_15;
      }
      v1 = v14;
      v25 = v15;
      v9 = v38;
      v26 = v14[12];
      if ( v26 )
      {
LABEL_30:
        v27 = (__int64)v36;
        do
        {
          while ( 1 )
          {
            v28 = *(_QWORD *)(v26 + 16);
            v29 = *(_QWORD *)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) >= v41 )
              break;
            v26 = *(_QWORD *)(v26 + 24);
            if ( !v29 )
              goto LABEL_34;
          }
          v27 = v26;
          v26 = *(_QWORD *)(v26 + 16);
        }
        while ( v28 );
LABEL_34:
        if ( (__int64 *)v27 != v36 && v41 >= *(_DWORD *)(v27 + 32) )
          goto LABEL_37;
        goto LABEL_36;
      }
LABEL_47:
      v27 = (__int64)v36;
LABEL_36:
      v42 = &v41;
      v27 = sub_359C130(v1 + 10, v27, &v42);
LABEL_37:
      *(_DWORD *)(v27 + 36) = v25;
      v30 = v9 + 40;
      *(_BYTE *)(v27 + 40) = v39;
      if ( v9 + 40 != v37 )
      {
        while ( 1 )
        {
          v9 = v30;
          if ( sub_2DADC00(v30) )
            break;
          v30 += 40;
          if ( v37 == v30 )
            goto LABEL_42;
        }
        if ( v37 != v30 )
          continue;
      }
      goto LABEL_42;
    }
    v31 = 1;
    while ( v21 != -4096 )
    {
      v32 = v31 + 1;
      v19 = (v18 - 1) & (v31 + v19);
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v16 == *v20 )
        goto LABEL_17;
      v31 = v32;
    }
LABEL_50:
    v23 = 0;
    goto LABEL_21;
  }
  return sub_35A8C50(v1);
}
