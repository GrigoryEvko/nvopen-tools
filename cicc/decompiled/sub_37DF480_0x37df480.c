// Function: sub_37DF480
// Address: 0x37df480
//
__int64 __fastcall sub_37DF480(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9

  if ( (unsigned __int16)(*(_WORD *)(a2 + 68) - 14) > 1u )
    return 0;
  v2 = sub_B10CD0(a2 + 56);
  if ( sub_35051D0((_QWORD *)(a1 + 128), v2) )
    return sub_37DEEC0(a1, a2, v3, v4, v5, v6);
  else
    return 1;
}
