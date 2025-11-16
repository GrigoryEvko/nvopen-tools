// Function: sub_1315920
// Address: 0x1315920
//
__int64 __fastcall sub_1315920(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v4; // r8
  __int64 v5; // rax

  if ( a1 && *(_QWORD *)(a1 + 144) )
  {
    a3 = (unsigned int)a3;
    v4 = *(unsigned __int8 *)(a1 + (unsigned int)a3 + 161);
    v5 = 224 * v4;
  }
  else
  {
    v5 = 0;
    LODWORD(v4) = 0;
    a3 = (unsigned int)a3;
  }
  if ( a4 )
    *a4 = v4;
  return dword_5060A40[a3] + a2 + v5;
}
