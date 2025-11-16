// Function: sub_1BAA5F0
// Address: 0x1baa5f0
//
__int64 __fastcall sub_1BAA5F0(unsigned __int64 a1, __int64 a2, int *a3, __int64 *a4, __int64 a5)
{
  _QWORD *v9; // rax
  __int64 v10; // rsi
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rcx

  v9 = (_QWORD *)sub_1B93320(a1, a2, a3);
  if ( v9
    || (v9 = (_QWORD *)sub_1BAA320(a1, a2, a3, (__int64)a4)) != 0
    || (v9 = (_QWORD *)sub_1B93400(a1, a2, a3)) != 0
    || (v9 = (_QWORD *)sub_1BA9FD0(a1, a2, a4)) != 0 )
  {
    v9[4] = a5;
    v10 = *(_QWORD *)(a5 + 112);
    v9[2] = a5 + 112;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    v9[1] = v10 | v9[1] & 7LL;
    *(_QWORD *)(v10 + 8) = v9 + 1;
    *(_QWORD *)(a5 + 112) = *(_QWORD *)(a5 + 112) & 7LL | (unsigned __int64)(v9 + 1);
    return 1;
  }
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    v12 = sub_22077B0(48);
    if ( v12 )
    {
      *(_BYTE *)(v12 + 24) = 8;
      v13 = 0;
      *(_QWORD *)(v12 + 40) = a2;
      *(_QWORD *)v12 = &unk_49F6F18;
    }
    else
    {
      v13 = MEMORY[8] & 7;
    }
    v14 = *(_QWORD *)(a5 + 112);
    *(_QWORD *)(v12 + 32) = a5;
    *(_QWORD *)(v12 + 16) = a5 + 112;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v12 + 8) = v14 | v13;
    *(_QWORD *)(v14 + 8) = v12 + 8;
    *(_QWORD *)(a5 + 112) = *(_QWORD *)(a5 + 112) & 7LL | (v12 + 8);
    return 1;
  }
  return sub_1B93750(a1, a2, a5, a3);
}
