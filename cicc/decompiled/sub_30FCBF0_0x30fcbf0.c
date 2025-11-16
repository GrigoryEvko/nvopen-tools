// Function: sub_30FCBF0
// Address: 0x30fcbf0
//
_QWORD *__fastcall sub_30FCBF0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  char v3; // dl
  _BYTE v5[352]; // [rsp+0h] [rbp-2F0h] BYREF
  __int64 v6; // [rsp+160h] [rbp-190h] BYREF
  _BYTE v7[352]; // [rsp+168h] [rbp-188h] BYREF

  v6 = a2;
  memset(v5, 0, sizeof(v5));
  qmemcpy(v7, v5, sizeof(v7));
  v2 = sub_30FCAD0((_QWORD *)(a1 + 120), (unsigned __int64 *)&v6);
  if ( v3 )
    qmemcpy(v2 + 5, (const void *)(sub_BC1CD0(*(_QWORD *)(a1 + 16), &unk_502ED90, a2) + 8), 0x160u);
  return v2 + 5;
}
