// Function: sub_2491640
// Address: 0x2491640
//
__int64 __fastcall sub_2491640(_QWORD *a1, __int64 a2)
{
  int v2; // eax
  __int64 *v3; // rdi
  __int64 v5; // rax
  __int64 v6; // [rsp+8h] [rbp-18h]

  v2 = *(unsigned __int8 *)(a2 + 8);
  switch ( (_BYTE)v2 )
  {
    case 2:
      v5 = 1;
      return (**(__int64 (__fastcall ***)(_QWORD, _QWORD))a1[v5])(a1[v5], *a1);
    case 3:
      v5 = 2;
      return (**(__int64 (__fastcall ***)(_QWORD, _QWORD))a1[v5])(a1[v5], *a1);
    case 4:
      v5 = 3;
      return (**(__int64 (__fastcall ***)(_QWORD, _QWORD))a1[v5])(a1[v5], *a1);
  }
  if ( (unsigned int)(v2 - 17) > 1 )
    return 0;
  if ( sub_BCEA30(a2) )
    return 0;
  v3 = (__int64 *)sub_2491640(a1, *(_QWORD *)(a2 + 24));
  if ( !v3 )
    return 0;
  BYTE4(v6) = *(_BYTE *)(a2 + 8) == 18;
  LODWORD(v6) = *(_DWORD *)(a2 + 32);
  return sub_BCE1B0(v3, v6);
}
