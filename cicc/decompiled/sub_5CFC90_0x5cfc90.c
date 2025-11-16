// Function: sub_5CFC90
// Address: 0x5cfc90
//
__int64 __fastcall sub_5CFC90(char a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // rbx

  v2 = sub_5CF860(a1, a2);
  if ( v2 && (v3 = v2, (unsigned int)sub_72AE80(*(_QWORD *)(v2[4] + 40))) )
    return *(_QWORD *)(*(_QWORD *)(v3[4] + 40) + 184LL);
  else
    return 0;
}
