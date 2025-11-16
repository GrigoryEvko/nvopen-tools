// Function: sub_FD5AF0
// Address: 0xfd5af0
//
void __fastcall sub_FD5AF0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  __int64 v4; // r15
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdi
  int v9; // eax
  int v10; // ecx
  __int64 v11; // rdi
  int v12; // edx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // eax
  int v16; // edx
  __int64 v17; // rdi
  int v18; // eax
  int v19; // edx
  __int64 v20; // [rsp+8h] [rbp-38h]
  __int64 v21; // [rsp+8h] [rbp-38h]

  v2 = *a2;
  v3 = *(_QWORD *)(*a2 + 16);
  if ( v3 )
  {
    v4 = *(_QWORD *)(v3 + 16);
    if ( v4 )
    {
      v6 = *(_QWORD *)(v4 + 16);
      if ( v6 )
      {
        if ( *(_QWORD *)(v6 + 16) )
        {
          sub_FD5AF0(a1, v6 + 16);
          v7 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 16LL);
          *(_DWORD *)(v7 + 64) = (*(_DWORD *)(v7 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v7 + 64) & 0xF8000000;
          v8 = *(_QWORD *)(v4 + 16);
          v9 = *(_DWORD *)(v8 + 64);
          v10 = (v9 + 0x7FFFFFF) & 0x7FFFFFF;
          *(_DWORD *)(v8 + 64) = v10 | v9 & 0xF8000000;
          if ( !v10 )
          {
            v21 = v7;
            sub_FD59A0(v8, a1);
            v7 = v21;
          }
          *(_QWORD *)(v4 + 16) = v7;
          v6 = *(_QWORD *)(*(_QWORD *)(v3 + 16) + 16LL);
        }
        *(_DWORD *)(v6 + 64) = (*(_DWORD *)(v6 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v6 + 64) & 0xF8000000;
        v11 = *(_QWORD *)(v3 + 16);
        v12 = *(_DWORD *)(v11 + 64);
        v13 = (v12 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v11 + 64) = v13 | v12 & 0xF8000000;
        if ( !v13 )
        {
          v20 = v6;
          sub_FD59A0(v11, a1);
          v6 = v20;
        }
        *(_QWORD *)(v3 + 16) = v6;
        v4 = *(_QWORD *)(*(_QWORD *)(v2 + 16) + 16LL);
      }
      *(_DWORD *)(v4 + 64) = (*(_DWORD *)(v4 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v4 + 64) & 0xF8000000;
      v14 = *(_QWORD *)(v2 + 16);
      v15 = *(_DWORD *)(v14 + 64);
      v16 = (v15 + 0x7FFFFFF) & 0x7FFFFFF;
      *(_DWORD *)(v14 + 64) = v16 | v15 & 0xF8000000;
      if ( !v16 )
        sub_FD59A0(v14, a1);
      *(_QWORD *)(v2 + 16) = v4;
      v3 = *(_QWORD *)(*a2 + 16);
    }
    *(_DWORD *)(v3 + 64) = (*(_DWORD *)(v3 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v3 + 64) & 0xF8000000;
    v17 = *a2;
    v18 = *(_DWORD *)(*a2 + 64);
    v19 = (v18 + 0x7FFFFFF) & 0x7FFFFFF;
    *(_DWORD *)(*a2 + 64) = v19 | v18 & 0xF8000000;
    if ( !v19 )
      sub_FD59A0(v17, a1);
    *a2 = v3;
  }
}
