// Function: sub_3735D90
// Address: 0x3735d90
//
__int64 __fastcall sub_3735D90(__int64 *a1, unsigned __int64 **a2, char a3, int a4, __int64 a5)
{
  __int64 v7; // rax
  int v8; // ebx
  int v9; // r8d
  int v10; // r9d
  __int64 v11; // rax
  __int64 v12; // r13
  bool v13; // zf

  v7 = sub_31DA930(a1[23]);
  v8 = sub_AE4380(v7, 0);
  v11 = sub_31DE8D0(a1[23], 0, 261, a4, v9, v10, a3);
  *(_DWORD *)(v11 + 32) = 2;
  v12 = v11;
  *(_BYTE *)(v11 + 36) = 1;
  v13 = *(_BYTE *)(v11 + 130) == 0;
  *(_BYTE *)(v11 + 128) = (v8 == 4) + 126;
  if ( v13 )
    *(_WORD *)(v11 + 129) = 257;
  else
    *(_BYTE *)(v11 + 129) = 1;
  sub_3249B00(a1, a2, 11, 237);
  sub_32499D0(a1, a2, 65549, 3);
  if ( sub_3734FE0((__int64)a1) )
    return sub_3249B00(a1, a2, 6, a5);
  else
    return sub_32493C0(a1, a2, 6, v12);
}
