// Function: sub_324C6D0
// Address: 0x324c6d0
//
__int64 __fastcall sub_324C6D0(_QWORD *a1, __int16 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // rax

  v6 = a1[11];
  a1[21] += 48LL;
  v7 = (v6 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( a1[12] >= (unsigned __int64)(v7 + 48) && v6 )
  {
    a1[11] = v7 + 48;
    if ( !v7 )
    {
      MEMORY[0x28] = 0;
      BUG();
    }
  }
  else
  {
    v7 = sub_9D1E70((__int64)(a1 + 11), 48, 48, 4);
  }
  *(_WORD *)(v7 + 28) = a2;
  *(_BYTE *)(v7 + 30) = 0;
  *(_QWORD *)v7 = v7 | 4;
  *(_QWORD *)(v7 + 8) = 0;
  *(_QWORD *)(v7 + 16) = 0;
  *(_DWORD *)(v7 + 24) = -1;
  *(_QWORD *)(v7 + 32) = 0;
  *(_QWORD *)(v7 + 40) = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v8 = *(_QWORD **)(a3 + 32);
  if ( v8 )
  {
    *(_QWORD *)v7 = *v8;
    **(_QWORD **)(a3 + 32) = v7 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)(a3 + 32) = v7;
  if ( a4 )
    sub_324C3F0((__int64)a1, a4, v7);
  return v7;
}
