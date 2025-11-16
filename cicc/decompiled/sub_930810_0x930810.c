// Function: sub_930810
// Address: 0x930810
//
__int64 __fastcall sub_930810(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // rax
  bool v6; // zf
  __int64 v7; // r13
  __int64 v8; // r9
  __int64 v9; // rax
  unsigned int v10; // r8d
  __int64 v11; // rdx
  __int64 v12; // r11
  unsigned __int8 v13; // dl
  __int64 v14; // rdi
  __int64 v15; // r13
  __int64 result; // rax
  __int64 v17; // [rsp+8h] [rbp-88h]
  __int64 v18; // [rsp+10h] [rbp-80h]
  unsigned int v19; // [rsp+1Ch] [rbp-74h]
  _QWORD *v20; // [rsp+20h] [rbp-70h]
  __int64 v21; // [rsp+28h] [rbp-68h]
  _QWORD v22[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v23; // [rsp+40h] [rbp-50h] BYREF
  __int64 v24; // [rsp+48h] [rbp-48h]
  _QWORD v25[8]; // [rsp+50h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v20 = v22;
  v21 = 0x200000000LL;
  v22[0] = sub_B9B140(v3, "llvm.loop.mustprogress", 22);
  v4 = *(_QWORD *)(a1 + 40);
  LODWORD(v21) = 1;
  v5 = sub_B9C770(v4, v22, 1, 0, 1);
  v6 = *(_QWORD *)(a2 + 48) == 0;
  v23 = v25;
  v7 = v5;
  v25[0] = 0;
  v24 = 0x200000001LL;
  if ( !v6 || (v9 = 1, (*(_BYTE *)(a2 + 7) & 0x20) != 0) )
  {
    v8 = sub_B91F50(a2, "llvm.loop", 9);
    v9 = (unsigned int)v24;
    if ( v8 )
    {
      v10 = 1;
      while ( 1 )
      {
        v13 = *(_BYTE *)(v8 - 16);
        if ( (v13 & 2) != 0 )
        {
          if ( v10 >= *(_DWORD *)(v8 - 24) )
            break;
          v11 = *(_QWORD *)(v8 - 32);
        }
        else
        {
          if ( v10 >= ((*(_WORD *)(v8 - 16) >> 6) & 0xFu) )
            break;
          v11 = v8 + -16 - 8LL * ((v13 >> 2) & 0xF);
        }
        v12 = *(_QWORD *)(v11 + 8LL * v10);
        if ( v9 + 1 > (unsigned __int64)HIDWORD(v24) )
        {
          v18 = *(_QWORD *)(v11 + 8LL * v10);
          v17 = v8;
          v19 = v10;
          sub_C8D5F0(&v23, v25, v9 + 1, 8);
          v9 = (unsigned int)v24;
          v8 = v17;
          v12 = v18;
          v10 = v19;
        }
        ++v10;
        v23[v9] = v12;
        v9 = (unsigned int)(v24 + 1);
        LODWORD(v24) = v24 + 1;
      }
    }
    if ( HIDWORD(v24) < (unsigned __int64)(v9 + 1) )
    {
      sub_C8D5F0(&v23, v25, v9 + 1, 8);
      v9 = (unsigned int)v24;
    }
  }
  v23[v9] = v7;
  v14 = *(_QWORD *)(a1 + 40);
  LODWORD(v24) = v24 + 1;
  v15 = sub_B9C770(v14, v23, (unsigned int)v24, 0, 1);
  sub_BA6610(v15, 0, v15);
  result = sub_B9A090(a2, "llvm.loop", 9, v15);
  if ( v23 != v25 )
    result = _libc_free(v23, "llvm.loop");
  if ( v20 != v22 )
    return _libc_free(v20, "llvm.loop");
  return result;
}
