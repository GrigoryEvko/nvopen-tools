// Function: sub_151A3A0
// Address: 0x151a3a0
//
__int64 __fastcall sub_151A3A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rdx
  _BYTE **v4; // r15
  __int64 v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v10; // [rsp+8h] [rbp-148h]
  _BYTE *v11; // [rsp+10h] [rbp-140h] BYREF
  __int64 v12; // [rsp+18h] [rbp-138h]
  _BYTE v13[304]; // [rsp+20h] [rbp-130h] BYREF

  v2 = *(unsigned int *)(a2 + 8);
  v11 = v13;
  v12 = 0x2000000000LL;
  if ( v2 <= 0x20 )
  {
    v3 = 8 * v2;
    v4 = (_BYTE **)(a2 - v3);
    if ( a2 != a2 - v3 )
      goto LABEL_3;
LABEL_10:
    v7 = (unsigned int)v12;
    goto LABEL_6;
  }
  sub_16CD150(&v11, v13, v2, 8);
  v4 = (_BYTE **)(a2 - 8LL * *(unsigned int *)(a2 + 8));
  if ( (_BYTE **)a2 == v4 )
    goto LABEL_10;
  do
  {
LABEL_3:
    v5 = sub_1519FE0(a1, *v4);
    v6 = (unsigned int)v12;
    if ( (unsigned int)v12 >= HIDWORD(v12) )
    {
      v10 = v5;
      sub_16CD150(&v11, v13, 0, 8);
      v6 = (unsigned int)v12;
      v5 = v10;
    }
    ++v4;
    *(_QWORD *)&v11[8 * v6] = v5;
    v7 = (unsigned int)(v12 + 1);
    LODWORD(v12) = v12 + 1;
  }
  while ( (_BYTE **)a2 != v4 );
LABEL_6:
  v8 = sub_1627350(*(_QWORD *)(a1 + 216), v11, v7, 0, 1);
  if ( v11 != v13 )
    _libc_free((unsigned __int64)v11);
  return v8;
}
