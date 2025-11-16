// Function: sub_16521E0
// Address: 0x16521e0
//
__int64 __fastcall sub_16521E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rax
  __int64 result; // rax

  v2 = *a1;
  if ( !*a1 )
    goto LABEL_4;
  sub_16E2CE0(a2, v2);
  v3 = *(_BYTE **)(v2 + 24);
  if ( (unsigned __int64)v3 < *(_QWORD *)(v2 + 16) )
  {
    *(_QWORD *)(v2 + 24) = v3 + 1;
    *v3 = 10;
LABEL_4:
    result = *((unsigned __int8 *)a1 + 74);
    *((_BYTE *)a1 + 73) = 1;
    *((_BYTE *)a1 + 72) |= result;
    return result;
  }
  sub_16E7DE0(v2, 10);
  result = *((unsigned __int8 *)a1 + 74);
  *((_BYTE *)a1 + 73) = 1;
  *((_BYTE *)a1 + 72) |= result;
  return result;
}
