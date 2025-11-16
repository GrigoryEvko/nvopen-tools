// Function: sub_1ECBF60
// Address: 0x1ecbf60
//
char __fastcall sub_1ECBF60(__int64 *a1, unsigned int a2)
{
  __int64 v2; // rdx
  int v3; // eax
  unsigned int v5; // [rsp+Ch] [rbp-4h] BYREF

  v2 = *a1;
  v5 = a2;
  v3 = *(_DWORD *)(*(_QWORD *)(v2 + 160) + 88LL * a2 + 16);
  switch ( v3 )
  {
    case 2:
      LOBYTE(v3) = sub_1ECB700(a1 + 7, &v5);
      break;
    case 3:
      LOBYTE(v3) = sub_1ECB700(a1 + 1, &v5);
      break;
    case 1:
      LOBYTE(v3) = sub_1ECB700(a1 + 13, &v5);
      break;
  }
  return v3;
}
