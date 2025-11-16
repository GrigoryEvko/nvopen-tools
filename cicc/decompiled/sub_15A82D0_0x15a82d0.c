// Function: sub_15A82D0
// Address: 0x15a82d0
//
unsigned __int64 __fastcall sub_15A82D0(__int64 a1, unsigned int a2, unsigned int a3, unsigned int a4, unsigned int a5)
{
  unsigned __int16 v5; // r12
  char *v8; // rax
  char *v9; // r14
  int v10; // ecx
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  unsigned __int64 *v15; // rsi
  char *v16; // rcx
  __int64 v17; // rax
  unsigned __int64 result; // rax
  char *v19; // r14
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+18h] [rbp-38h] BYREF
  unsigned __int64 v23; // [rsp+20h] [rbp-30h]

  if ( a5 > 0xFFFFFF )
    sub_16BD130("Invalid bit width, must be a 24bit integer", 1);
  v5 = a3;
  if ( (a3 & 0xFFFF0000) != 0 )
    sub_16BD130("Invalid ABI alignment, must be a 16bit integer", 1);
  if ( (a4 & 0xFFFF0000) != 0 )
    sub_16BD130("Invalid preferred alignment, must be a 16bit integer", 1);
  if ( a3 )
  {
    if ( (a3 & (a3 - 1LL)) != 0 )
      sub_16BD130("Invalid ABI alignment, must be a power of 2", 1);
    if ( !a4 )
      goto LABEL_23;
    if ( (a4 & (a4 - 1)) != 0 )
      goto LABEL_25;
    if ( a3 > a4 )
LABEL_23:
      sub_16BD130("Preferred alignment cannot be less than the ABI alignment", 1);
  }
  else if ( a4 && (a4 & (a4 - 1)) != 0 )
  {
LABEL_25:
    sub_16BD130("Invalid preferred alignment, must be a power of 2", 1);
  }
  v8 = (char *)sub_15A8270(a1, a2, a5);
  v9 = v8;
  if ( v8 != (char *)(*(_QWORD *)(a1 + 48) + 8LL * *(unsigned int *)(a1 + 56))
    && (unsigned __int8)*v8 == a2
    && (result = *(_DWORD *)v8 >> 8, (_DWORD)result == a5) )
  {
    *((_WORD *)v9 + 2) = v5;
    *((_WORD *)v9 + 3) = a4;
  }
  else
  {
    v10 = a5;
    v21 = a1 + 48;
    v14 = sub_15A8080(a2, v5, a4, v10);
    v11 = *(unsigned int *)(a1 + 56);
    v12 = *(_QWORD *)(a1 + 48);
    v22 = v14;
    v13 = 8 * v11;
    LODWORD(v14) = v11;
    v15 = (unsigned __int64 *)(v12 + 8 * v11);
    if ( v9 == (char *)v15 )
    {
      if ( (unsigned int)v11 >= *(_DWORD *)(a1 + 60) )
      {
        sub_16CD150(v21, a1 + 64, 0, 8);
        v15 = (unsigned __int64 *)(*(_QWORD *)(a1 + 48) + 8LL * *(unsigned int *)(a1 + 56));
      }
      result = v22;
      *v15 = v22;
      ++*(_DWORD *)(a1 + 56);
    }
    else
    {
      if ( v11 >= *(unsigned int *)(a1 + 60) )
      {
        v19 = &v9[-v12];
        sub_16CD150(v21, a1 + 64, 0, 8);
        v12 = *(_QWORD *)(a1 + 48);
        v14 = *(unsigned int *)(a1 + 56);
        v13 = 8 * v14;
        v9 = &v19[v12];
        v15 = (unsigned __int64 *)(v12 + 8 * v14);
      }
      v16 = (char *)(v12 + v13 - 8);
      if ( v15 )
      {
        *v15 = *(_QWORD *)v16;
        v12 = *(_QWORD *)(a1 + 48);
        v14 = *(unsigned int *)(a1 + 56);
        v13 = 8 * v14;
        v16 = (char *)(v12 + 8 * v14 - 8);
      }
      if ( v9 != v16 )
      {
        memmove((void *)(v12 + v13 - (v16 - v9)), v9, v16 - v9);
        LODWORD(v14) = *(_DWORD *)(a1 + 56);
      }
      v17 = (unsigned int)(v14 + 1);
      *(_DWORD *)(a1 + 56) = v17;
      if ( v9 <= (char *)&v22 && (unsigned __int64)&v22 < *(_QWORD *)(a1 + 48) + 8 * v17 )
      {
        result = v23;
        *(_QWORD *)v9 = v23;
      }
      else
      {
        result = v22;
        *(_QWORD *)v9 = v22;
      }
    }
  }
  return result;
}
