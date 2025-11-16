// Function: sub_2F3FD50
// Address: 0x2f3fd50
//
__int64 __fastcall sub_2F3FD50(__int64 a1, __int64 *a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  bool v6; // zf
  __int64 v7; // rdi
  const char *v9; // [rsp+0h] [rbp-50h] BYREF
  char v10; // [rsp+20h] [rbp-30h]
  char v11; // [rsp+21h] [rbp-2Fh]

  v3 = *a2;
  v4 = sub_22077B0(0x18u);
  v5 = v4;
  if ( v4 )
  {
    v6 = *(_BYTE *)(a1 + 188) == 0;
    *(_QWORD *)(v4 + 8) = v3;
    *(_DWORD *)(v4 + 16) = 0;
    *(_QWORD *)v4 = off_4A2AD30;
    if ( !v6 )
    {
      v11 = 1;
      v9 = "Requested regalloc eviction advisor analysis could not be created. Using default";
      v10 = 3;
      sub_B6ECE0(v3, (__int64)&v9);
    }
  }
  v7 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v5;
  if ( v7 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
  return 0;
}
