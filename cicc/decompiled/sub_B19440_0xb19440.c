// Function: sub_B19440
// Address: 0xb19440
//
void __fastcall sub_B19440(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  _QWORD *v4; // rdi
  int v5; // ebx
  _QWORD **v6; // r8
  unsigned int v7; // eax
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // r9
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  int v13; // r13d
  _QWORD *v14; // rcx
  __int64 *v15; // rdx
  _QWORD *v16; // rsi
  __int64 v17; // [rsp+0h] [rbp-250h]
  _QWORD **v18; // [rsp+8h] [rbp-248h]
  _QWORD *v19; // [rsp+10h] [rbp-240h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-238h]
  unsigned int v21; // [rsp+1Ch] [rbp-234h]
  _QWORD v22[70]; // [rsp+20h] [rbp-230h] BYREF

  if ( *(_BYTE *)(a1 + 112) )
  {
    *(_DWORD *)(a1 + 116) = 0;
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 96);
    v21 = 32;
    v19 = v22;
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      *(_DWORD *)(v2 + 72) = 0;
      v4 = v22;
      v5 = 1;
      v22[0] = v2;
      v6 = &v19;
      v7 = 1;
      v22[1] = v3;
      v20 = 1;
      do
      {
        while ( 1 )
        {
          v13 = v5++;
          v14 = &v4[2 * v7 - 2];
          v16 = (_QWORD *)*v14;
          v15 = (__int64 *)v14[1];
          if ( v15 != (__int64 *)(*(_QWORD *)(*v14 + 24LL) + 8LL * *(unsigned int *)(*v14 + 32LL)) )
            break;
          --v7;
          *((_DWORD *)v16 + 19) = v13;
          v20 = v7;
          if ( !v7 )
            goto LABEL_9;
        }
        v8 = *v15;
        v14[1] = v15 + 1;
        v9 = v20;
        v10 = *(_QWORD *)(v8 + 24);
        v11 = v20 + 1LL;
        if ( v11 > v21 )
        {
          v16 = v22;
          v17 = *(_QWORD *)(v8 + 24);
          v18 = v6;
          sub_C8D5F0(v6, v22, v11, 16);
          v4 = v19;
          v9 = v20;
          v10 = v17;
          v6 = v18;
        }
        v12 = &v4[2 * v9];
        *v12 = v8;
        v4 = v19;
        v12[1] = v10;
        LODWORD(v12) = v20;
        *(_DWORD *)(v8 + 72) = v13;
        v7 = (_DWORD)v12 + 1;
        v20 = v7;
      }
      while ( v7 );
LABEL_9:
      *(_DWORD *)(a1 + 116) = 0;
      *(_BYTE *)(a1 + 112) = 1;
      if ( v4 != v22 )
        _libc_free(v4, v16);
    }
  }
}
