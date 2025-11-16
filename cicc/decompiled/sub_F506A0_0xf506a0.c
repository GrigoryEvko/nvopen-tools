// Function: sub_F506A0
// Address: 0xf506a0
//
__int64 __fastcall sub_F506A0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  char v3; // r15
  __int64 v4; // rax
  __int64 v5; // rdx
  char v6; // r13
  unsigned __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  unsigned int v10; // r15d
  __int64 v11; // rax
  _BYTE *v12; // rax
  unsigned __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+8h] [rbp-48h]
  unsigned __int8 v16; // [rsp+10h] [rbp-40h]

  v2 = sub_B43CC0(a2);
  v3 = sub_AE5020(v2, a1);
  v4 = sub_9208B0(v2, a1);
  v15 = v5;
  v6 = v5;
  v14 = v4;
  v7 = 8 * (((1LL << v3) + ((unsigned __int64)(v4 + 7) >> 3) - 1) >> v3 << v3);
  v8 = sub_AF4940(
         *(_QWORD *)(*(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL),
         *(_QWORD *)(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) + 24LL));
  v15 = v9;
  v10 = (unsigned __int8)v9;
  v14 = v8;
  if ( (_BYTE)v9 )
    goto LABEL_11;
  v11 = *(_QWORD *)(a2 - 32);
  if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  if ( *(_DWORD *)(v11 + 36) == 69 )
  {
    v12 = (_BYTE *)sub_B58EB0(a2, 0);
    if ( v12 )
    {
      if ( *v12 == 60 )
      {
        sub_B4CFC0((__int64)&v14, (__int64)v12, v2);
        v10 = v16;
        if ( v16 )
        {
          if ( v6 || (v10 = 0, !(_BYTE)v15) )
LABEL_11:
            LOBYTE(v10) = v14 <= v7;
        }
      }
    }
  }
  return v10;
}
