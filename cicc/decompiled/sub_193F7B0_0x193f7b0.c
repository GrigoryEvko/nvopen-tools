// Function: sub_193F7B0
// Address: 0x193f7b0
//
__int64 __fastcall sub_193F7B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 result; // rax
  char v9; // r12
  __int64 v10; // rbx
  __int64 v12; // r14
  __int64 v13; // r8
  int v14; // r9d
  char v15; // dl
  bool v16; // r10
  unsigned int v17; // edx
  _QWORD *v18; // r15
  _QWORD *v19; // rsi
  unsigned int v20; // edi
  _QWORD *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // rsi
  int v26; // edi
  unsigned int i; // eax
  __int64 *v28; // r10
  unsigned int v29; // eax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  bool v33; // [rsp+0h] [rbp-90h]
  bool v34; // [rsp+0h] [rbp-90h]
  bool v35; // [rsp+0h] [rbp-90h]
  __int64 *v36; // [rsp+0h] [rbp-90h]
  bool v37; // [rsp+0h] [rbp-90h]
  unsigned __int64 v38; // [rsp+8h] [rbp-88h]
  __int64 v41; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-68h]
  __int64 v43; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v44; // [rsp+38h] [rbp-58h]
  __int64 v45; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v46; // [rsp+48h] [rbp-48h]
  char v47; // [rsp+50h] [rbp-40h]

  v3 = sub_146F1B0(*(_QWORD *)(a1 + 32), a2);
  v4 = *(_QWORD *)(a1 + 32);
  v5 = v3;
  v6 = sub_1456040(v3);
  v7 = sub_145CF80(v4, v6, 0, 0);
  result = sub_147A340(v4, 0x27u, v5, v7);
  if ( *(_QWORD *)(a2 + 8) )
  {
    v9 = result;
    v10 = a1 + 88;
    v12 = *(_QWORD *)(a2 + 8);
    v38 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
    while ( 1 )
    {
      v18 = sub_1648700(v12);
      result = *(_QWORD *)(a1 + 96);
      if ( *(_QWORD *)(a1 + 104) == result )
      {
        v19 = (_QWORD *)(result + 8LL * *(unsigned int *)(a1 + 116));
        v20 = *(_DWORD *)(a1 + 116);
        if ( (_QWORD *)result != v19 )
        {
          v21 = 0;
          while ( v18 != *(_QWORD **)result )
          {
            if ( *(_QWORD *)result == -2 )
              v21 = (_QWORD *)result;
            result += 8;
            if ( v19 == (_QWORD *)result )
            {
              if ( !v21 )
                goto LABEL_28;
              *v21 = v18;
              v16 = v9;
              --*(_DWORD *)(a1 + 120);
              ++*(_QWORD *)(a1 + 88);
              if ( v9 )
                goto LABEL_5;
              goto LABEL_19;
            }
          }
          goto LABEL_9;
        }
LABEL_28:
        if ( v20 < *(_DWORD *)(a1 + 112) )
          break;
      }
      result = (__int64)sub_16CCBA0(v10, (__int64)v18);
      if ( v15 )
        goto LABEL_4;
LABEL_9:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        return result;
    }
    *(_DWORD *)(a1 + 116) = v20 + 1;
    *v19 = v18;
    ++*(_QWORD *)(a1 + 88);
LABEL_4:
    v16 = v9;
    if ( !v9 )
    {
LABEL_19:
      v22 = *(unsigned int *)(a1 + 584);
      if ( !(_DWORD)v22 )
        goto LABEL_26;
      v23 = *(_QWORD *)(a1 + 568);
      v24 = (((((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4) | v38)
            - 1
            - ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4) | v38)
           - 1
           - ((unsigned __int64)(((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4)) << 32));
      v25 = ((9 * (((v24 - 1 - (v24 << 13)) >> 8) ^ (v24 - 1 - (v24 << 13)))) >> 15)
          ^ (9 * (((v24 - 1 - (v24 << 13)) >> 8) ^ (v24 - 1 - (v24 << 13))));
      v26 = 1;
      for ( i = (v22 - 1) & (((v25 - 1 - (v25 << 27)) >> 31) ^ (v25 - 1 - ((_DWORD)v25 << 27))); ; i = (v22 - 1) & v29 )
      {
        v28 = (__int64 *)(v23 + 48LL * i);
        v13 = *v28;
        if ( *v28 == a2 && v18 == (_QWORD *)v28[1] )
          break;
        if ( v13 == -8 && v28[1] == -8 )
          goto LABEL_26;
        v29 = v26 + i;
        ++v26;
      }
      if ( v28 == (__int64 *)(48 * v22 + v23) )
      {
LABEL_26:
        v17 = *(_DWORD *)(a1 + 264);
        v16 = 0;
        if ( v17 < *(_DWORD *)(a1 + 268) )
          goto LABEL_6;
        goto LABEL_27;
      }
      v47 = 1;
      v44 = *((_DWORD *)v28 + 6);
      if ( v44 > 0x40 )
      {
        v36 = (__int64 *)(v23 + 48LL * i);
        sub_16A4FD0((__int64)&v43, (const void **)v28 + 2);
        v28 = v36;
      }
      else
      {
        v43 = v28[2];
      }
      v46 = *((_DWORD *)v28 + 10);
      if ( v46 > 0x40 )
      {
        sub_16A4FD0((__int64)&v45, (const void **)v28 + 4);
        v16 = v47;
      }
      else
      {
        v30 = v28[4];
        v16 = v47;
        v45 = v30;
      }
      if ( v16 )
      {
        sub_158ACE0((__int64)&v41, (__int64)&v43);
        v31 = 1LL << ((unsigned __int8)v42 - 1);
        if ( v42 > 0x40 )
        {
          v32 = *(_QWORD *)(v41 + 8LL * ((v42 - 1) >> 6)) & v31;
          v16 = v32 == 0;
          if ( v41 )
          {
            v37 = v32 == 0;
            j_j___libc_free_0_0(v41);
            v16 = v37;
          }
        }
        else
        {
          v16 = (v41 & v31) == 0;
        }
        if ( v47 )
        {
          if ( v46 > 0x40 && v45 )
          {
            v34 = v16;
            j_j___libc_free_0_0(v45);
            v16 = v34;
          }
          if ( v44 > 0x40 && v43 )
          {
            v35 = v16;
            j_j___libc_free_0_0(v43);
            v16 = v35;
          }
        }
      }
    }
LABEL_5:
    v17 = *(_DWORD *)(a1 + 264);
    if ( v17 < *(_DWORD *)(a1 + 268) )
    {
LABEL_6:
      result = *(_QWORD *)(a1 + 256) + 32LL * v17;
      if ( result )
      {
        *(_QWORD *)(result + 8) = v18;
        *(_BYTE *)(result + 24) = v16;
        *(_QWORD *)result = a2;
        *(_QWORD *)(result + 16) = a3;
        v17 = *(_DWORD *)(a1 + 264);
      }
      *(_DWORD *)(a1 + 264) = v17 + 1;
      goto LABEL_9;
    }
LABEL_27:
    v33 = v16;
    sub_16CD150(a1 + 256, (const void *)(a1 + 272), 0, 32, v13, v14);
    v17 = *(_DWORD *)(a1 + 264);
    v16 = v33;
    goto LABEL_6;
  }
  return result;
}
