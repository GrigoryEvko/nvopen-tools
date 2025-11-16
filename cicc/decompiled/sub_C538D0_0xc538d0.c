// Function: sub_C538D0
// Address: 0xc538d0
//
__int64 __fastcall sub_C538D0(_QWORD *a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rbx
  size_t v8; // r15
  const void *v9; // r14
  unsigned int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  _QWORD *v14; // rcx
  __int64 v15; // rax
  char v16; // r13
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 result; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned int v22; // eax
  int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // r8d
  _QWORD *v26; // rcx
  _QWORD *v27; // r13
  __int64 v28; // rax
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  __int64 v30; // [rsp+10h] [rbp-70h]
  unsigned int v31; // [rsp+10h] [rbp-70h]
  _QWORD *v32; // [rsp+18h] [rbp-68h]
  _QWORD v33[4]; // [rsp+20h] [rbp-60h] BYREF
  char v34; // [rsp+40h] [rbp-40h]
  char v35; // [rsp+41h] [rbp-3Fh]

  v6 = a3;
  v7 = a2;
  v8 = *(_QWORD *)(a2 + 32);
  if ( v8 )
  {
    v32 = (_QWORD *)(a3 + 128);
    if ( (*(_BYTE *)(a2 + 13) & 0x20) != 0 )
    {
      v21 = *(_QWORD *)(a2 + 24);
      v30 = *(_QWORD *)(a3 + 128) + 8LL * *(unsigned int *)(a3 + 136);
      v22 = sub_C92610(v21, v8);
      v23 = sub_C92860(v32, v21, v8, v22);
      if ( v23 == -1 )
        result = *(_QWORD *)(v6 + 128) + 8LL * *(unsigned int *)(v6 + 136);
      else
        result = *(_QWORD *)(v6 + 128) + 8LL * v23;
      if ( v30 != result )
        return result;
      v8 = *(_QWORD *)(a2 + 32);
    }
    v9 = *(const void **)(a2 + 24);
    v33[1] = v8;
    v33[0] = v9;
    v10 = sub_C92610(v9, v8);
    v12 = (unsigned int)sub_C92740(v32, v9, v8, v10);
    v14 = (_QWORD *)(*(_QWORD *)(v6 + 128) + 8 * v12);
    if ( *v14 )
    {
      if ( *v14 != -8 )
      {
        v15 = sub_CEADF0(v32, v9, v11, v14, v12, v13);
        v16 = 1;
        v17 = sub_CB6200(v15, *a1, a1[1]);
        v18 = sub_904010(v17, ": CommandLine Error: Option '");
        a2 = (__int64)"' registered more than once!\n";
        a1 = (_QWORD *)sub_A51340(v18, *(const void **)(v7 + 24), *(_QWORD *)(v7 + 32));
        sub_904010((__int64)a1, "' registered more than once!\n");
        if ( ((*(_WORD *)(v7 + 12) >> 7) & 3) == 1 )
          goto LABEL_6;
LABEL_12:
        if ( (*(_BYTE *)(v7 + 13) & 8) != 0 )
        {
          result = *(unsigned int *)(v6 + 88);
          a4 = *(unsigned int *)(v6 + 92);
          if ( result + 1 > a4 )
          {
            a2 = v6 + 96;
            a1 = (_QWORD *)(v6 + 80);
            sub_C8D5F0(v6 + 80, v6 + 96, result + 1, 8);
            result = *(unsigned int *)(v6 + 88);
          }
          a3 = *(_QWORD *)(v6 + 80);
          *(_QWORD *)(a3 + 8 * result) = v7;
          ++*(_DWORD *)(v6 + 88);
        }
        else
        {
          result = *(_BYTE *)(v7 + 12) & 7;
          if ( (_BYTE)result == 4 )
          {
            if ( *(_QWORD *)(v6 + 152) )
            {
              v28 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
              a2 = (__int64)v33;
              a1 = (_QWORD *)v7;
              v35 = 1;
              v33[0] = "Cannot specify more than one option with cl::ConsumeAfter!";
              v34 = 3;
              sub_C53280(v7, (__int64)v33, 0, 0, v28);
              *(_QWORD *)(v6 + 152) = v7;
              goto LABEL_9;
            }
            *(_QWORD *)(v6 + 152) = v7;
          }
        }
        if ( !v16 )
          return result;
LABEL_9:
        v20 = sub_CEADF0(a1, a2, a3, a4, a5, a6);
        return sub_904010(v20, "inconsistency in registered CommandLine options");
      }
      --*(_DWORD *)(v6 + 144);
    }
    v29 = v14;
    v31 = v12;
    v24 = sub_C7D670(v8 + 17, 8);
    v25 = v31;
    v26 = v29;
    v27 = (_QWORD *)v24;
    if ( v8 )
    {
      memcpy((void *)(v24 + 16), v9, v8);
      v25 = v31;
      v26 = v29;
    }
    *((_BYTE *)v27 + v8 + 16) = 0;
    a1 = v32;
    a2 = v25;
    *v27 = v8;
    v27[1] = v7;
    *v26 = v27;
    v16 = 0;
    ++*(_DWORD *)(v6 + 140);
    sub_C929D0(v32, v25);
  }
  else
  {
    v16 = 0;
  }
  if ( ((*(_WORD *)(v7 + 12) >> 7) & 3) != 1 )
    goto LABEL_12;
LABEL_6:
  result = *(unsigned int *)(v6 + 40);
  a4 = *(unsigned int *)(v6 + 44);
  if ( result + 1 > a4 )
  {
    a2 = v6 + 48;
    a1 = (_QWORD *)(v6 + 32);
    sub_C8D5F0(v6 + 32, v6 + 48, result + 1, 8);
    result = *(unsigned int *)(v6 + 40);
  }
  a3 = *(_QWORD *)(v6 + 32);
  *(_QWORD *)(a3 + 8 * result) = v7;
  ++*(_DWORD *)(v6 + 40);
  if ( v16 )
    goto LABEL_9;
  return result;
}
