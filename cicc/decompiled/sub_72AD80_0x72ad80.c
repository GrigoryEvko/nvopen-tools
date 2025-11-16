// Function: sub_72AD80
// Address: 0x72ad80
//
__int64 __fastcall sub_72AD80(_QWORD *a1)
{
  int v1; // esi
  _BYTE *v2; // r12
  __int64 result; // rax
  int v4[5]; // [rsp+Ch] [rbp-14h] BYREF

  v4[0] = 0;
  sub_7296C0(v4);
  if ( !*a1 || (v1 = *(_DWORD *)(*(_QWORD *)(*a1 + 96LL) + 96LL), v1 == -1) )
    v1 = sub_880E90();
  v2 = sub_726EB0(6, v1, 0);
  sub_729730(v4[0]);
  *((_QWORD *)v2 + 2) = a1[5];
  result = a1[21];
  *(_QWORD *)(result + 152) = v2;
  *((_QWORD *)v2 + 4) = a1;
  return result;
}
