// Function: sub_2EB3C30
// Address: 0x2eb3c30
//
void __fastcall sub_2EB3C30(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdi
  int v5; // ebx
  _QWORD *v6; // r8
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r9
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // r13d
  _QWORD *v14; // rcx
  __int64 *v15; // rdx
  __int64 v16; // [rsp+0h] [rbp-250h]
  _QWORD *v17; // [rsp+8h] [rbp-248h]
  _QWORD *v18; // [rsp+10h] [rbp-240h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-238h]
  unsigned int v20; // [rsp+1Ch] [rbp-234h]
  _QWORD v21[70]; // [rsp+20h] [rbp-230h] BYREF

  if ( *(_BYTE *)(a1 + 136) )
  {
    *(_DWORD *)(a1 + 140) = 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 120);
    v20 = 32;
    v18 = v21;
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      *(_DWORD *)(v2 + 72) = 0;
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
          v15 = (__int64 *)v14[1];
          if ( v15 != (__int64 *)(*(_QWORD *)(*v14 + 24LL) + 8LL * *(unsigned int *)(*v14 + 32LL)) )
            break;
          --v7;
          *(_DWORD *)(*v14 + 76LL) = v13;
          v19 = v7;
          if ( !v7 )
            goto LABEL_9;
        }
        v8 = *v15;
        v14[1] = v15 + 1;
        v9 = v19;
        v10 = *(_QWORD *)(v8 + 24);
        v11 = v19 + 1LL;
        if ( v11 > v20 )
        {
          v16 = *(_QWORD *)(v8 + 24);
          v17 = v6;
          sub_C8D5F0((__int64)v6, v21, v11, 0x10u, (__int64)v6, v10);
          v4 = v18;
          v9 = v19;
          v10 = v16;
          v6 = v17;
        }
        v12 = &v4[2 * v9];
        *v12 = v8;
        v4 = v18;
        v12[1] = v10;
        LODWORD(v12) = v19;
        *(_DWORD *)(v8 + 72) = v13;
        v7 = (_DWORD)v12 + 1;
        v19 = v7;
      }
      while ( v7 );
LABEL_9:
      *(_DWORD *)(a1 + 140) = 0;
      *(_BYTE *)(a1 + 136) = 1;
      if ( v4 != v21 )
        _libc_free((unsigned __int64)v4);
    }
  }
}
