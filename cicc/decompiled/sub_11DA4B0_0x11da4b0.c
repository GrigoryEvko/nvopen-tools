// Function: sub_11DA4B0
// Address: 0x11da4b0
//
char __fastcall sub_11DA4B0(__int64 a1, int *a2, __int64 a3)
{
  int *v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  int v8; // r12d
  __int64 *v9; // rax
  int v10; // r12d
  __int64 *v11; // rax
  int *v13; // [rsp+8h] [rbp-48h]
  unsigned int v14[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v4 = a2;
  v5 = sub_B491C0(a1);
  if ( v5 )
  {
    v6 = v5;
    v5 = (__int64)&a2[a3];
    v13 = (int *)v5;
    if ( (int *)v5 != a2 )
    {
      do
      {
        v14[0] = *v4;
        if ( !(unsigned __int8)sub_B49B80(a1, v14[0], 40) )
        {
          v10 = v14[0];
          v11 = (__int64 *)sub_BD5C60(a1);
          *(_QWORD *)(a1 + 72) = sub_A7A090((__int64 *)(a1 + 72), v11, v10 + 1, 40);
        }
        if ( !(unsigned __int8)sub_B49B80(a1, v14[0], 43) )
        {
          v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (v14[0] - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 8LL);
          if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
            v7 = **(_QWORD **)(v7 + 16);
          LOBYTE(v5) = sub_B2F070(v6, *(_DWORD *)(v7 + 8) >> 8);
          if ( (_BYTE)v5 )
            goto LABEL_10;
          v8 = v14[0];
          v9 = (__int64 *)sub_BD5C60(a1);
          *(_QWORD *)(a1 + 72) = sub_A7A090((__int64 *)(a1 + 72), v9, v8 + 1, 43);
        }
        LOBYTE(v5) = sub_11DA2E0(a1, v14, 1, 1u);
LABEL_10:
        ++v4;
      }
      while ( v13 != v4 );
    }
  }
  return v5;
}
