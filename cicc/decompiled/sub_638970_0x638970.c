// Function: sub_638970
// Address: 0x638970
//
__int64 __fastcall sub_638970(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v5; // rdi
  char v6; // dl
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 result; // rax
  int v10; // eax
  __int64 v11; // [rsp-10h] [rbp-30h]
  _QWORD v12[3]; // [rsp+8h] [rbp-18h] BYREF

  if ( a2 )
    *(_BYTE *)(a1 + 176) |= 0x20u;
  v5 = *(_QWORD *)(a1 + 288);
  *(_BYTE *)(a1 + 176) |= 8u;
  v6 = *(_BYTE *)(v5 + 140);
  v12[0] = *(_QWORD *)&dword_4F063F8;
  if ( v6 == 12 )
  {
    v7 = v5;
    do
    {
      v7 = *(_QWORD *)(v7 + 160);
      v6 = *(_BYTE *)(v7 + 140);
    }
    while ( v6 == 12 );
  }
  *(_DWORD *)(a1 + 176) = *(_DWORD *)(a1 + 176) & 0xFFF6FDFF | ((v6 == 0) << 9) | 0x80000;
  if ( dword_4F077C4 != 2 )
  {
    if ( (*(_BYTE *)(a1 + 176) & 2) != 0 )
    {
      *(_DWORD *)(a1 + 176) |= 0x400004u;
    }
    else if ( !unk_4D0421C )
    {
      *(_DWORD *)(a1 + 176) |= 0x400004u;
    }
  }
  sub_637180(v5, a2, (__m128i *)(a1 + 136), (_QWORD *)a1, 1u, a3, v12);
  v8 = *(_QWORD *)(a1 + 144);
  result = v11;
  if ( v8 )
  {
    v10 = *(unsigned __int8 *)(v8 + 48);
    *(_BYTE *)(v8 + 50) |= 0x20u;
    result = v10 & 0xFFFFFFFB;
    if ( (_BYTE)result == 2 )
    {
      *(_BYTE *)(*(_QWORD *)(v8 + 56) + 171LL) |= 1u;
      result = sub_8D23E0(*(_QWORD *)(a1 + 288));
      if ( !(_DWORD)result )
      {
        result = *(_QWORD *)(v8 + 56);
        *(_QWORD *)(result + 128) = *(_QWORD *)(a1 + 288);
      }
    }
  }
  return result;
}
