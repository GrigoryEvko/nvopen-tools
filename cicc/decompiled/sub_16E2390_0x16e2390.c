// Function: sub_16E2390
// Address: 0x16e2390
//
void __fastcall sub_16E2390(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  unsigned __int64 v7; // rbx
  char *v8; // rax
  size_t v9; // rdx
  __int64 v10; // r9
  char *v11; // r10
  char *v12; // rcx
  _DWORD *v13; // r8
  size_t v14; // r13
  int v15; // eax
  char *s1; // [rsp+8h] [rbp-48h]
  char *v18; // [rsp+10h] [rbp-40h]
  __int64 v19; // [rsp+18h] [rbp-38h]

  v5 = sub_16E2000((__int64 *)a1);
  v7 = v6;
  s1 = (char *)v5;
  v18 = (char *)v5;
  v19 = v6;
  v8 = sub_16E0C80(*(_DWORD *)(a1 + 44));
  v10 = v19;
  v11 = v18;
  v12 = s1;
  v13 = a4;
  if ( v9 > v7 || (v14 = v9) != 0 && (v15 = memcmp(s1, v8, v9), v12 = s1, v13 = a4, v10 = v19, v11 = v18, v15) )
  {
    if ( *(_DWORD *)(a1 + 44) == 11 && v7 > 4 && *(_DWORD *)v12 == 1868783981 && v12[4] == 115 )
    {
      v10 = v7 - 5;
      v11 = v12 + 5;
    }
  }
  else
  {
    v11 = &v12[v14];
    v10 = v7 - v14;
  }
  sub_16DDD70(v11, v10, a2, a3, v13);
}
