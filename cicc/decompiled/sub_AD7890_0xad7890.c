// Function: sub_AD7890
// Address: 0xad7890
//
bool __fastcall sub_AD7890(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  _BYTE *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  _BYTE *v11; // rbx
  _BYTE *v12; // rbx

  if ( *(_BYTE *)a1 == 18 )
  {
    if ( *(_QWORD *)(a1 + 24) == sub_C33340(a1, a2, a3, a4, a5) )
      v5 = *(_QWORD *)(a1 + 32);
    else
      v5 = a1 + 24;
    return (*(_BYTE *)(v5 + 20) & 7) == 3;
  }
  else if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17 <= 1
         && (v7 = sub_AD7630(a1, 0, a3), (v11 = v7) != 0)
         && *v7 == 18 )
  {
    if ( *((_QWORD *)v7 + 3) == sub_C33340(a1, 0, v8, v9, v10) )
      v12 = (_BYTE *)*((_QWORD *)v11 + 4);
    else
      v12 = v11 + 24;
    return (v12[20] & 7) == 3;
  }
  else
  {
    return sub_AC30F0(a1);
  }
}
