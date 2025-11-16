// Function: sub_2649280
// Address: 0x2649280
//
void __fastcall sub_2649280(char *a1, char *a2, __int64 a3)
{
  char *v4; // r15
  __int64 v5; // r13
  char *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdx
  bool v9; // dl
  __int64 v10; // rdi
  char *v11; // rbx
  __int64 v12; // r12
  __int64 v13; // rcx
  __int64 v14; // rdx
  volatile signed __int32 *v15; // r14
  signed __int32 v16; // edx
  signed __int32 v17; // edx
  volatile signed __int32 *v18; // rdi
  __int64 v19; // rdi
  char *v20; // rbx
  char *v21; // r8
  __int64 v22; // rax
  char *v23; // r15
  __int64 v24; // rsi
  __int64 v25; // rcx
  bool v26; // cl
  __int64 v27; // rdx
  volatile signed __int32 *v28; // r14
  signed __int32 v29; // eax
  char *v30; // r14
  char *v31; // rdx
  signed __int32 v32; // eax
  unsigned int v33; // esi
  unsigned int *v34; // rcx
  unsigned int *v35; // rdi
  unsigned int *v36; // r9
  unsigned int v37; // ebx
  __int64 v38; // [rsp+10h] [rbp-A0h]
  __int64 v39; // [rsp+10h] [rbp-A0h]
  char *v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  char *v43; // [rsp+28h] [rbp-88h]
  char *v45; // [rsp+38h] [rbp-78h]
  _QWORD v46[4]; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v47[2]; // [rsp+60h] [rbp-50h] BYREF
  unsigned int *v48; // [rsp+70h] [rbp-40h]

  if ( a1 != a2 && a2 != a1 + 16 )
  {
    v4 = a1 + 32;
    do
    {
      v5 = *((_QWORD *)v4 - 2);
      v45 = v4;
      v6 = v4 - 16;
      if ( !*(_DWORD *)(v5 + 40) )
        goto LABEL_26;
      if ( !*(_DWORD *)(*(_QWORD *)a1 + 40LL) )
        goto LABEL_9;
      v7 = *(unsigned __int8 *)(v5 + 16);
      v8 = *(unsigned __int8 *)(*(_QWORD *)a1 + 16LL);
      if ( (_BYTE)v7 == (_BYTE)v8 )
      {
        sub_22B0690(v46, (__int64 *)(v5 + 24));
        v37 = *(_DWORD *)v46[2];
        sub_22B0690(v47, (__int64 *)(*(_QWORD *)a1 + 24LL));
        v5 = *((_QWORD *)v4 - 2);
        v6 = v4 - 16;
        v9 = v37 < *v48;
      }
      else
      {
        v9 = *(_DWORD *)(a3 + 4 * v7) < *(_DWORD *)(a3 + 4 * v8);
      }
      if ( v9 )
      {
LABEL_9:
        v10 = *((_QWORD *)v4 - 1);
        *((_QWORD *)v4 - 1) = 0;
        *((_QWORD *)v4 - 2) = 0;
        v42 = v10;
        if ( v6 - a1 > 0 )
        {
          v38 = a3;
          v11 = v6;
          v12 = (v6 - a1) >> 4;
          do
          {
            while ( 1 )
            {
              v13 = *((_QWORD *)v11 - 2);
              v14 = *((_QWORD *)v11 - 1);
              v11 -= 16;
              *((_QWORD *)v11 + 1) = 0;
              v15 = (volatile signed __int32 *)*((_QWORD *)v11 + 3);
              *(_QWORD *)v11 = 0;
              *((_QWORD *)v11 + 3) = v14;
              *((_QWORD *)v11 + 2) = v13;
              if ( v15 )
              {
                if ( &_pthread_key_create )
                {
                  v16 = _InterlockedExchangeAdd(v15 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v16 = *((_DWORD *)v15 + 2);
                  *((_DWORD *)v15 + 2) = v16 - 1;
                }
                if ( v16 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
                  if ( &_pthread_key_create )
                  {
                    v17 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v17 = *((_DWORD *)v15 + 3);
                    *((_DWORD *)v15 + 3) = v17 - 1;
                  }
                  if ( v17 == 1 )
                    break;
                }
              }
              if ( !--v12 )
                goto LABEL_20;
            }
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
            --v12;
          }
          while ( v12 );
LABEL_20:
          a3 = v38;
        }
        v18 = (volatile signed __int32 *)*((_QWORD *)a1 + 1);
        *(_QWORD *)a1 = v5;
        *((_QWORD *)a1 + 1) = v42;
        if ( !v18 )
          goto LABEL_23;
      }
      else
      {
LABEL_26:
        v19 = *((_QWORD *)v4 - 1);
        *((_QWORD *)v4 - 2) = 0;
        v20 = v4 - 32;
        v21 = v4;
        *((_QWORD *)v4 - 1) = 0;
        v39 = v19;
        if ( !*(_DWORD *)(v5 + 40) )
        {
          *(_QWORD *)v6 = v5;
          *((_QWORD *)v6 + 1) = v19;
          goto LABEL_23;
        }
        while ( 1 )
        {
          v22 = *(_QWORD *)v20;
          v23 = v20;
          if ( *(_DWORD *)(*(_QWORD *)v20 + 40LL) )
          {
            v24 = *(unsigned __int8 *)(v5 + 16);
            v25 = *(unsigned __int8 *)(v22 + 16);
            if ( (_BYTE)v24 == (_BYTE)v25 )
            {
              v43 = v21;
              sub_22B0690(v47, (__int64 *)(v5 + 24));
              v21 = v43;
              v22 = *(_QWORD *)v20;
              if ( *(_DWORD *)(*(_QWORD *)v20 + 40LL) )
              {
                v34 = *(unsigned int **)(v22 + 32);
                v35 = &v34[*(unsigned int *)(v22 + 48)];
                v33 = *v34;
                if ( v34 != v35 )
                {
                  while ( 1 )
                  {
                    v33 = *v34;
                    v36 = v34;
                    if ( *v34 <= 0xFFFFFFFD )
                      break;
                    if ( v35 == ++v34 )
                    {
                      v33 = v36[1];
                      break;
                    }
                  }
                }
              }
              else
              {
                v33 = *(_DWORD *)(*(_QWORD *)(v22 + 32) + 4LL * *(unsigned int *)(v22 + 48));
              }
              v26 = v33 > *v48;
            }
            else
            {
              v26 = *(_DWORD *)(a3 + 4 * v24) < *(_DWORD *)(a3 + 4 * v25);
            }
            if ( !v26 )
              break;
          }
          v27 = *((_QWORD *)v20 + 1);
          v28 = (volatile signed __int32 *)*((_QWORD *)v20 + 3);
          *((_QWORD *)v20 + 2) = v22;
          *((_QWORD *)v20 + 1) = 0;
          *(_QWORD *)v20 = 0;
          *((_QWORD *)v20 + 3) = v27;
          if ( v28 )
          {
            if ( &_pthread_key_create )
            {
              v29 = _InterlockedExchangeAdd(v28 + 2, 0xFFFFFFFF);
            }
            else
            {
              v29 = *((_DWORD *)v28 + 2);
              *((_DWORD *)v28 + 2) = v29 - 1;
            }
            if ( v29 == 1 )
            {
              v41 = v21;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v28 + 16LL))(v28);
              v21 = v41;
              if ( &_pthread_key_create )
              {
                v32 = _InterlockedExchangeAdd(v28 + 3, 0xFFFFFFFF);
              }
              else
              {
                v32 = *((_DWORD *)v28 + 3);
                *((_DWORD *)v28 + 3) = v32 - 1;
              }
              if ( v32 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v28 + 24LL))(v28);
                v21 = v41;
              }
            }
          }
          v20 -= 16;
          if ( !*(_DWORD *)(v5 + 40) )
          {
            v30 = v23;
            v4 = v21;
            v18 = (volatile signed __int32 *)*((_QWORD *)v30 + 1);
            v31 = v30;
            goto LABEL_37;
          }
        }
        v18 = (volatile signed __int32 *)*((_QWORD *)v20 + 3);
        v31 = v20 + 16;
        v4 = v21;
LABEL_37:
        *(_QWORD *)v31 = v5;
        *((_QWORD *)v31 + 1) = v39;
        if ( !v18 )
          goto LABEL_23;
      }
      sub_A191D0(v18);
LABEL_23:
      v4 += 16;
    }
    while ( a2 != v45 );
  }
}
