// Function: sub_D72860
// Address: 0xd72860
//
__int64 __fastcall sub_D72860(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // esi
  __int64 v13; // rdi
  int v14; // esi
  __int64 v15; // rcx
  __int64 *v16; // rax
  __int64 v17; // r8
  __int64 v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // r15
  unsigned int i; // r12d
  __int64 v22; // r9
  int v23; // esi
  __int64 v24; // rdi
  int v25; // esi
  unsigned int v26; // ecx
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdi
  __int64 v30; // rcx
  int v31; // esi
  _BYTE *v32; // rsi
  __int64 v33; // rdx
  _QWORD *v34; // rbx
  __int64 result; // rax
  _QWORD *v36; // r12
  int v37; // eax
  int v38; // edx
  int v39; // [rsp+10h] [rbp-1F0h]
  unsigned int v40; // [rsp+14h] [rbp-1ECh]
  __int64 v41; // [rsp+18h] [rbp-1E8h]
  _QWORD v42[4]; // [rsp+20h] [rbp-1E0h] BYREF
  _BYTE *v43; // [rsp+40h] [rbp-1C0h] BYREF
  __int64 v44; // [rsp+48h] [rbp-1B8h]
  _BYTE v45[432]; // [rsp+50h] [rbp-1B0h] BYREF

  v6 = a2 + 24;
  v8 = *(_QWORD *)(a2 + 40);
  v9 = v8 + 48;
  if ( a2 + 24 != v8 + 48 )
  {
    do
    {
      v10 = v6;
      v6 = *(_QWORD *)(v6 + 8);
      v11 = v10 - 24;
      v12 = *(_DWORD *)(*a1 + 56);
      v13 = *(_QWORD *)(*a1 + 40);
      if ( v12 )
      {
        v14 = v12 - 1;
        v15 = v14 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = (__int64 *)(v13 + 16 * v15);
        v17 = *v16;
        if ( v11 == *v16 )
        {
LABEL_4:
          v18 = v16[1];
          if ( v18 )
            sub_D6E4B0(a1, v18, 0, v15, v17, a6);
        }
        else
        {
          v37 = 1;
          while ( v17 != -4096 )
          {
            a6 = (unsigned int)(v37 + 1);
            v15 = v14 & (unsigned int)(v37 + v15);
            v16 = (__int64 *)(v13 + 16LL * (unsigned int)v15);
            v17 = *v16;
            if ( v11 == *v16 )
              goto LABEL_4;
            v37 = a6;
          }
        }
      }
    }
    while ( v9 != v6 );
  }
  v43 = v45;
  v44 = 0x1000000000LL;
  v19 = *(_QWORD *)(v8 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9 == v19 )
    goto LABEL_40;
  if ( !v19 )
    BUG();
  v20 = v19 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA || (v39 = sub_B46E30(v20)) == 0 )
  {
LABEL_40:
    v32 = v45;
    v33 = 0;
  }
  else
  {
    for ( i = 0; i != v39; ++i )
    {
      v41 = sub_B46EC0(v20, i);
      sub_D6D880(a1, v8, v41);
      v23 = *(_DWORD *)(*a1 + 56);
      v24 = *(_QWORD *)(*a1 + 40);
      if ( v23 )
      {
        v25 = v23 - 1;
        v26 = v25 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
        v27 = v24 + 16LL * v26;
        v28 = *(_QWORD *)v27;
        if ( v41 == *(_QWORD *)v27 )
        {
LABEL_14:
          v29 = *(_QWORD *)(v27 + 8);
          if ( v29 )
          {
            v30 = 0;
            v31 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
            if ( v31 )
            {
              do
              {
                while ( 1 )
                {
                  v27 = (unsigned int)v30;
                  if ( v8 == *(_QWORD *)(*(_QWORD *)(v29 - 8)
                                       + 32LL * *(unsigned int *)(v29 + 76)
                                       + 8LL * (unsigned int)v30) )
                    break;
                  v30 = (unsigned int)(v30 + 1);
                  if ( (_DWORD)v30 == v31 )
                    goto LABEL_20;
                }
                v40 = v30;
                sub_D68A80(v29, v30);
                v30 = v40;
                v31 = *(_DWORD *)(v29 + 4) & 0x7FFFFFF;
              }
              while ( v40 != v31 );
            }
LABEL_20:
            v42[0] = 4;
            v42[1] = 0;
            v42[2] = v29;
            if ( v29 != -4096 && v29 != -8192 )
              sub_BD73F0((__int64)v42);
            sub_D6B260((__int64)&v43, (char *)v42, v27, v30, v28, v22);
            sub_D68D70(v42);
          }
        }
        else
        {
          v38 = 1;
          while ( v28 != -4096 )
          {
            v22 = (unsigned int)(v38 + 1);
            v26 = v25 & (v38 + v26);
            v27 = v24 + 16LL * v26;
            v28 = *(_QWORD *)v27;
            if ( v41 == *(_QWORD *)v27 )
              goto LABEL_14;
            v38 = v22;
          }
        }
      }
    }
    v32 = v43;
    v33 = (unsigned int)v44;
  }
  sub_D6FF00((__int64)a1, (__int64)v32, v33);
  v34 = v43;
  result = 3LL * (unsigned int)v44;
  v36 = &v43[24 * (unsigned int)v44];
  if ( v43 != (_BYTE *)v36 )
  {
    do
    {
      v36 -= 3;
      result = sub_D68D70(v36);
    }
    while ( v34 != v36 );
    v36 = v43;
  }
  if ( v36 != (_QWORD *)v45 )
    return _libc_free(v36, v32);
  return result;
}
