// Function: sub_1001990
// Address: 0x1001990
//
__int64 __fastcall sub_1001990(_QWORD **a1)
{
  int v1; // edx
  __int64 *v2; // rax
  __int64 v4; // [rsp-10h] [rbp-10h]

  v1 = *((unsigned __int8 *)a1 + 8);
  if ( (unsigned int)(v1 - 17) > 1 )
    return sub_BCB2A0(*a1);
  BYTE4(v4) = (_BYTE)v1 == 18;
  LODWORD(v4) = *((_DWORD *)a1 + 8);
  v2 = (__int64 *)sub_BCB2A0(*a1);
  return sub_BCE1B0(v2, v4);
}
