// Function: sub_9877A0
// Address: 0x9877a0
//
__int64 __fastcall sub_9877A0(__int64 a1, unsigned __int8 *a2)
{
  int v3; // eax
  __int64 result; // rax
  int v5; // eax
  __int64 v6; // rdi
  unsigned __int8 *v7; // rdx
  char v8; // dl
  char v9; // dl
  unsigned __int8 *v10; // rbx
  __int64 v11; // [rsp+10h] [rbp-20h]
  char v12; // [rsp+18h] [rbp-18h]

  v3 = *a2;
  if ( (unsigned __int8)v3 > 0x1Cu )
  {
    v5 = v3 - 29;
  }
  else
  {
    if ( (_BYTE)v3 != 5 )
      return 0;
    v5 = *((unsigned __int16 *)a2 + 1);
  }
  if ( v5 != 47 )
    return 0;
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v11 = sub_9208B0(v6, *(_QWORD *)(*(_QWORD *)v7 + 8LL));
  v12 = v8;
  if ( sub_9208B0(*(_QWORD *)(a1 + 8), *((_QWORD *)a2 + 1)) != v11 || v9 != v12 )
    return 0;
  v10 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  result = 1;
  if ( *(_QWORD *)v10 != *(_QWORD *)(a1 + 16) )
    return 0;
  return result;
}
