// Function: sub_16E22F0
// Address: 0x16e22f0
//
void __fastcall sub_16E22F0(__int64 a1, _DWORD *a2, _DWORD *a3, _DWORD *a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  size_t v7; // r12
  __int64 v8; // r15
  char *v9; // rax
  size_t v10; // rdx
  char *v11; // r9
  _DWORD *v12; // r8
  char *v13; // rcx
  size_t v14; // rbx
  int v15; // eax
  char *s1; // [rsp+8h] [rbp-48h]
  _DWORD *s1a; // [rsp+8h] [rbp-48h]
  char *v19; // [rsp+10h] [rbp-40h]
  char *v20; // [rsp+18h] [rbp-38h]

  v5 = sub_16E2120((__int64 *)a1);
  v7 = v6;
  s1 = (char *)v5;
  v8 = v6;
  v20 = (char *)v5;
  v9 = sub_16E0EC0(*(_DWORD *)(a1 + 48));
  v11 = v20;
  v12 = a4;
  if ( v10 <= v7 )
  {
    v13 = s1;
    v14 = v10;
    if ( !v10 || (s1a = a4, v19 = v13, v15 = memcmp(v13, v9, v10), v13 = v19, v12 = s1a, v11 = v20, !v15) )
    {
      v11 = &v13[v14];
      v8 = v7 - v14;
    }
  }
  sub_16DDD70(v11, v8, a2, a3, v12);
}
