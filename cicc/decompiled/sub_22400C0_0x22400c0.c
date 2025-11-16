// Function: sub_22400C0
// Address: 0x22400c0
//
_BOOL8 __fastcall sub_22400C0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // r13
  size_t v6; // r12
  _BYTE *v7; // rbp
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v11; // rax
  void *v12; // rdi
  char v13; // [rsp+4h] [rbp-54h]
  size_t v14[8]; // [rsp+18h] [rbp-40h] BYREF

  v4 = a1 + 120;
  v13 = a3;
  sub_222DF20(a1 + 120);
  *(_WORD *)(a1 + 344) = 0;
  *(_QWORD *)a1 = qword_4A07108;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 120) = &unk_4A07130;
  *(_QWORD *)(a1 + 8) = 0;
  sub_222DD70(a1 + 120, 0);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 120) = off_4A071A0;
  *(_QWORD *)a1 = off_4A07178;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 16) = off_4A07480;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  sub_220A990((volatile signed __int32 **)(a1 + 72));
  v6 = *(_QWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 80) = 0;
  v7 = *(_BYTE **)a2;
  *(_QWORD *)(a1 + 16) = off_4A07080;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  if ( &v7[v6] && !v7 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  v14[0] = v6;
  if ( v6 > 0xF )
  {
    v11 = sub_22409D0(a1 + 88, v14, 0);
    *(_QWORD *)(a1 + 88) = v11;
    v12 = (void *)v11;
    *(_QWORD *)(a1 + 104) = v14[0];
  }
  else
  {
    if ( v6 == 1 )
    {
      *(_BYTE *)(a1 + 104) = *v7;
      v8 = a1 + 104;
      goto LABEL_6;
    }
    if ( !v6 )
    {
      v8 = a1 + 104;
      goto LABEL_6;
    }
    v12 = (void *)(a1 + 104);
  }
  memcpy(v12, v7, v6);
  v6 = v14[0];
  v8 = *(_QWORD *)(a1 + 88);
LABEL_6:
  *(_QWORD *)(a1 + 96) = v6;
  v9 = 0;
  *(_BYTE *)(v8 + v6) = 0;
  *(_DWORD *)(a1 + 80) = a3 | 8;
  if ( (v13 & 3) != 0 )
    v9 = *(_QWORD *)(a1 + 96);
  sub_223FD50(a1 + 16, *(_QWORD *)(a1 + 88), 0, v9);
  return sub_222DD70(v4, a1 + 16);
}
