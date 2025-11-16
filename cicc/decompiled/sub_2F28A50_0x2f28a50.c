// Function: sub_2F28A50
// Address: 0x2f28a50
//
__int64 __fastcall sub_2F28A50(__int64 a1, __int64 a2, __int64 a3)
{
  if ( (unsigned __int8)sub_2F28740(a3) )
  {
    sub_2EAFFB0(a1);
    return a1;
  }
  else
  {
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    return a1;
  }
}
