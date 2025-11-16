// Function: sub_1E052A0
// Address: 0x1e052a0
//
void __fastcall sub_1E052A0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdi
  int v5; // ebx
  _QWORD *v6; // r8
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r15
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  int v13; // r13d
  _QWORD *v14; // rcx
  __int64 v15; // r9
  __int64 *v16; // rdx
  _QWORD *v17; // [rsp+8h] [rbp-248h]
  _QWORD *v18; // [rsp+10h] [rbp-240h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-238h]
  unsigned int v20; // [rsp+1Ch] [rbp-234h]
  _QWORD v21[70]; // [rsp+20h] [rbp-230h] BYREF

  if ( *(_BYTE *)(a1 + 72) )
  {
    *(_DWORD *)(a1 + 76) = 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 56);
    v20 = 32;
    v18 = v21;
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      *(_DWORD *)(v2 + 48) = 0;
      v4 = v21;
      v5 = 1;
      v21[0] = v2;
      v6 = &v18;
      v7 = 1;
      v21[1] = v3;
      v19 = 1;
      do
      {
        while ( 1 )
        {
          v13 = v5++;
          v14 = &v4[2 * v7 - 2];
          v15 = *v14;
          v16 = (__int64 *)v14[1];
          if ( *(__int64 **)(*v14 + 32LL) != v16 )
            break;
          --v7;
          *(_DWORD *)(v15 + 52) = v13;
          v19 = v7;
          if ( !v7 )
            goto LABEL_9;
        }
        v8 = *v16;
        v14[1] = v16 + 1;
        v9 = v19;
        v10 = *(_QWORD *)(v8 + 24);
        if ( v19 >= v20 )
        {
          v17 = v6;
          sub_16CD150((__int64)v6, v21, 0, 16, (int)v6, v15);
          v4 = v18;
          v9 = v19;
          v6 = v17;
        }
        v11 = &v4[2 * v9];
        *v11 = v8;
        v11[1] = v10;
        v12 = v19;
        *(_DWORD *)(v8 + 48) = v13;
        v4 = v18;
        v7 = v12 + 1;
        v19 = v7;
      }
      while ( v7 );
LABEL_9:
      *(_DWORD *)(a1 + 76) = 0;
      *(_BYTE *)(a1 + 72) = 1;
      if ( v4 != v21 )
        _libc_free((unsigned __int64)v4);
    }
  }
}
