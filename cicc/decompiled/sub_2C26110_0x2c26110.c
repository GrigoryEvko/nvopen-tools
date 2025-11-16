// Function: sub_2C26110
// Address: 0x2c26110
//
__int64 __fastcall sub_2C26110(__int64 a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int16 v5; // [rsp+70h] [rbp-B8h]
  __int16 v6; // [rsp+A0h] [rbp-88h]

  result = a1;
  v3 = *a2;
  v4 = a2[2];
  v6 = *((_WORD *)a2 + 12);
  v5 = *((_WORD *)a2 + 4);
  if ( *a2 != v4 )
  {
    do
    {
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v3 - 8) + 8LL) - 1 <= 1 )
        break;
      v3 -= 8;
    }
    while ( v4 != v3 );
  }
  *(_QWORD *)a1 = v3;
  *(_QWORD *)(a1 + 16) = v4;
  *(_WORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 40) = v4;
  *(_WORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 56) = v4;
  *(_WORD *)(a1 + 48) = v6;
  *(_WORD *)(a1 + 64) = v6;
  return result;
}
