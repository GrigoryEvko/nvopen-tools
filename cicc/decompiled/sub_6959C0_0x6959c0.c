// Function: sub_6959C0
// Address: 0x6959c0
//
__int64 __fastcall sub_6959C0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // r13
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned int v16; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v17[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = 0;
  if ( qword_4D03C50 )
    v2 = ((*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0) << 14;
  result = sub_72AA80(a1);
  if ( (_DWORD)result )
  {
    v17[0] = sub_724DC0(a1, a2, v4, v5, v6, v7);
    sub_72A510(a1, v17[0]);
    sub_7296C0(&v16);
    sub_740190(v17[0], a1, v2 | 0x20u);
    sub_729730(v16);
    return sub_724E30(v17);
  }
  else if ( *(_BYTE *)(a1 + 173) == 12 && *(_BYTE *)(a1 + 176) == 1 )
  {
    result = sub_731AB0(*(_QWORD *)(a1 + 184));
    if ( (_DWORD)result )
    {
      v8 = *(_QWORD *)(a1 + 184);
      v9 = *(_QWORD *)(*(_QWORD *)(sub_731B30() + 56) + 48LL);
      sub_72B840(v9);
      sub_7296B0(*(unsigned int *)(v9 + 164), a2, v10, v11);
      v12 = v2 | 0x2000u;
      v13 = sub_73B8B0(v8, v12);
      sub_7296B0(unk_4F073B8, v12, v14, v15);
      result = sub_72D910(v13, 3, a1);
      *(_QWORD *)(a1 + 184) = 0;
    }
  }
  return result;
}
