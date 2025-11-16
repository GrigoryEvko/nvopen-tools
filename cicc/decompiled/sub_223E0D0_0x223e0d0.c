// Function: sub_223E0D0
// Address: 0x223e0d0
//
__int64 *__fastcall sub_223E0D0(__int64 *a1, const char *a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  char *v7; // rbp
  __int64 v8; // rcx
  __int64 v9; // rcx
  __int64 v10; // rbx
  unsigned __int8 v11; // r14
  _QWORD *v12; // rdi
  unsigned __int8 *v13; // rax
  char *v14; // rbx
  __int64 v15; // rdi
  __int64 v17; // rax
  _DWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rbx
  unsigned __int8 v23; // r13
  _QWORD *v24; // rdi
  unsigned __int8 *v25; // rax
  __int64 v26; // rax
  __int64 (__fastcall *v27)(__int64, unsigned int); // rax
  _BYTE *v28; // r14
  __int64 v29; // rax
  __int64 (__fastcall *v30)(__int64, unsigned int); // rax
  __int64 v31; // [rsp+8h] [rbp-60h]
  int v32; // [rsp+14h] [rbp-54h]
  _BYTE *v33; // [rsp+18h] [rbp-50h]
  _BYTE v34[8]; // [rsp+20h] [rbp-48h] BYREF
  _QWORD *v35; // [rsp+28h] [rbp-40h]

  sub_223DFF0((__int64)v34, a1);
  if ( v34[0] )
  {
    v6 = *(_QWORD *)(*a1 - 24);
    v7 = (char *)a1 + v6;
    v8 = *(__int64 *)((char *)a1 + v6 + 16);
    v31 = v8;
    if ( a3 < v8 )
    {
      v32 = *((_DWORD *)v7 + 6) & 0xB0;
      if ( v32 != 32 )
      {
        v9 = v8 - a3;
        v10 = v9;
        if ( v7[225] )
        {
          v11 = v7[224];
        }
        else
        {
          v33 = (_BYTE *)*((_QWORD *)v7 + 30);
          if ( !v33 )
            sub_426219(v34, a1, v5, v9);
          if ( v33[56] )
          {
            v11 = v33[89];
          }
          else
          {
            sub_2216D60((__int64)v33);
            v11 = 32;
            v27 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v33 + 48LL);
            if ( v27 != sub_CE72A0 )
              v11 = v27((__int64)v33, 32u);
          }
          v7[224] = v11;
          v26 = *a1;
          v7[225] = 1;
          v6 = *(_QWORD *)(v26 - 24);
        }
        do
        {
          v12 = *(_QWORD **)((char *)a1 + v6 + 232);
          v13 = (unsigned __int8 *)v12[5];
          if ( (unsigned __int64)v13 < v12[6] )
          {
            *v13 = v11;
            ++v12[5];
          }
          else if ( (*(unsigned int (__fastcall **)(_QWORD *, _QWORD))(*v12 + 104LL))(v12, v11) == -1 )
          {
            sub_222DC80((__int64)a1 + *(_QWORD *)(*a1 - 24), *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 32) | 1);
            v7 = (char *)a1 + *(_QWORD *)(*a1 - 24);
            goto LABEL_11;
          }
          v6 = *(_QWORD *)(*a1 - 24);
          --v10;
        }
        while ( v10 );
        v7 = (char *)a1 + v6;
        if ( *(_DWORD *)((char *)a1 + v6 + 32) )
          goto LABEL_12;
        goto LABEL_23;
      }
LABEL_11:
      if ( !*((_DWORD *)v7 + 8) )
      {
LABEL_23:
        v18 = (_DWORD *)*((_QWORD *)v7 + 29);
        if ( a3 != (*(__int64 (__fastcall **)(_DWORD *, const char *, __int64))(*(_QWORD *)v18 + 96LL))(v18, a2, a3) )
        {
          v18 = (_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24));
          a2 = (const char *)(v18[8] | 1u);
          sub_222DC80((__int64)v18, (int)a2);
        }
        v20 = *(_QWORD *)(*a1 - 24);
        v7 = (char *)a1 + v20;
        if ( v32 == 32 )
        {
          v21 = *((unsigned int *)v7 + 8);
          if ( !(_DWORD)v21 )
          {
            v22 = v31 - a3;
            if ( v7[225] )
            {
              v23 = v7[224];
            }
            else
            {
              v28 = (_BYTE *)*((_QWORD *)v7 + 30);
              if ( !v28 )
                sub_426219(v18, a2, v21, v19);
              if ( v28[56] )
              {
                v23 = v28[89];
              }
              else
              {
                sub_2216D60(*((_QWORD *)v7 + 30));
                v23 = 32;
                v30 = *(__int64 (__fastcall **)(__int64, unsigned int))(*(_QWORD *)v28 + 48LL);
                if ( v30 != sub_CE72A0 )
                  v23 = v30((__int64)v28, 32u);
              }
              v7[224] = v23;
              v29 = *a1;
              v7[225] = 1;
              v20 = *(_QWORD *)(v29 - 24);
            }
            do
            {
              v24 = *(_QWORD **)((char *)a1 + v20 + 232);
              v25 = (unsigned __int8 *)v24[5];
              if ( (unsigned __int64)v25 < v24[6] )
              {
                *v25 = v23;
                ++v24[5];
              }
              else if ( (*(unsigned int (__fastcall **)(_QWORD *, _QWORD))(*v24 + 104LL))(v24, v23) == -1 )
              {
                sub_222DC80(
                  (__int64)a1 + *(_QWORD *)(*a1 - 24),
                  *(_DWORD *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 32) | 1);
                goto LABEL_21;
              }
              v20 = *(_QWORD *)(*a1 - 24);
              --v22;
            }
            while ( v22 );
            v7 = (char *)a1 + v20;
          }
        }
      }
LABEL_12:
      *((_QWORD *)v7 + 2) = 0;
      goto LABEL_13;
    }
    v17 = (*(__int64 (__fastcall **)(_QWORD, const char *, __int64))(**((_QWORD **)v7 + 29) + 96LL))(
            *((_QWORD *)v7 + 29),
            a2,
            a3);
    v7 = (char *)a1 + *(_QWORD *)(*a1 - 24);
    if ( a3 == v17 )
      goto LABEL_12;
    sub_222DC80((__int64)a1 + *(_QWORD *)(*a1 - 24), *((_DWORD *)v7 + 8) | 1);
LABEL_21:
    *(__int64 *)((char *)a1 + *(_QWORD *)(*a1 - 24) + 16) = 0;
  }
LABEL_13:
  v14 = (char *)v35 + *(_QWORD *)(*v35 - 24LL);
  if ( (v14[25] & 0x20) != 0 && !(unsigned __int8)sub_2252910() )
  {
    v15 = *((_QWORD *)v14 + 29);
    if ( v15 )
    {
      if ( (*(unsigned int (__fastcall **)(__int64))(*(_QWORD *)v15 + 48LL))(v15) == -1 )
        sub_222DC80(
          (__int64)v35 + *(_QWORD *)(*v35 - 24LL),
          *(_DWORD *)((char *)v35 + *(_QWORD *)(*v35 - 24LL) + 32) | 1);
    }
  }
  return a1;
}
