// Function: sub_222C290
// Address: 0x222c290
//
__int64 __fastcall sub_222C290(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rdi
  int v6; // r14d
  int v7; // eax
  bool v8; // al
  bool v9; // r12
  __off64_t v10; // r8
  __int64 v12; // rbx
  bool v13; // zf
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18[8]; // [rsp+8h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a1 + 200);
  if ( v5 )
  {
    v6 = 0;
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 40LL))(v5);
    if ( v7 >= 0 )
      v6 = v7;
    v8 = v7 <= 0;
  }
  else
  {
    v8 = 1;
    v6 = 0;
  }
  v9 = v8 && a2 != 0;
  if ( !sub_2207CD0((_QWORD *)(a1 + 104)) || v9 )
    return -1;
  if ( a3 != 1
    || a2
    || *(_BYTE *)(a1 + 170)
    && !(*(unsigned __int8 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 48LL))(*(_QWORD *)(a1 + 200)) )
  {
    if ( *(_BYTE *)(a1 + 192) )
    {
      v13 = *(_QWORD *)(a1 + 16) == *(_QWORD *)(a1 + 8);
      *(_BYTE *)(a1 + 192) = 0;
      v14 = *(_QWORD *)(a1 + 184);
      v15 = *(_QWORD *)(a1 + 152);
      v16 = *(_QWORD *)(a1 + 176) + !v13;
      *(_QWORD *)(a1 + 176) = v16;
      *(_QWORD *)(a1 + 8) = v15;
      *(_QWORD *)(a1 + 16) = v16;
      *(_QWORD *)(a1 + 24) = v14;
    }
    v17 = a2 * v6;
    v13 = *(_BYTE *)(a1 + 169) == 0;
    v18[0] = *(_QWORD *)(a1 + 124);
    if ( !v13 && a3 == 1 )
    {
      v18[0] = *(_QWORD *)(a1 + 140);
      v17 += (int)sub_222BE20(a1, (__int64)v18);
    }
    return sub_222BFB0(a1, v17, a3, v18[0]);
  }
  else
  {
    v12 = 0;
    v13 = *(_BYTE *)(a1 + 169) == 0;
    v18[0] = *(_QWORD *)(a1 + 124);
    if ( !v13 )
    {
      v18[0] = *(_QWORD *)(a1 + 140);
      v12 = (int)sub_222BE20(a1, (__int64)v18);
    }
    if ( *(_BYTE *)(a1 + 170) )
      v12 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
    v10 = sub_2207F40((FILE **)(a1 + 104), 0, 1);
    if ( v10 != -1 )
      v10 += v12;
  }
  return v10;
}
