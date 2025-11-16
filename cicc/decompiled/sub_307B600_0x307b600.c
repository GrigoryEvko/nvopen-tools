// Function: sub_307B600
// Address: 0x307b600
//
__int64 __fastcall sub_307B600(_QWORD *a1)
{
  unsigned __int64 v2; // r12
  __int64 v3; // rsi
  _QWORD *v4; // rbx
  _QWORD *v5; // r13
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi

  v2 = a1[25];
  *a1 = &unk_4A31918;
  if ( v2 )
  {
    sub_C7D6A0(*(_QWORD *)(v2 + 304), 16LL * *(unsigned int *)(v2 + 320), 8);
    sub_C7D6A0(*(_QWORD *)(v2 + 272), 4LL * *(unsigned int *)(v2 + 288), 4);
    sub_C7D6A0(*(_QWORD *)(v2 + 240), 4LL * *(unsigned int *)(v2 + 256), 4);
    sub_C7D6A0(*(_QWORD *)(v2 + 208), 4LL * *(unsigned int *)(v2 + 224), 4);
    sub_C7D6A0(*(_QWORD *)(v2 + 152), 16LL * *(unsigned int *)(v2 + 168), 8);
    v3 = *(unsigned int *)(v2 + 136);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v2 + 120);
      v5 = &v4[2 * v3];
      do
      {
        if ( *v4 != -8192 && *v4 != -4096 )
        {
          v6 = v4[1];
          if ( v6 )
          {
            v7 = *(_QWORD *)(v6 + 96);
            if ( v7 != v6 + 112 )
              _libc_free(v7);
            v8 = *(_QWORD *)(v6 + 24);
            if ( v8 != v6 + 40 )
              _libc_free(v8);
            j_j___libc_free_0(v6);
          }
        }
        v4 += 2;
      }
      while ( v5 != v4 );
      v3 = *(unsigned int *)(v2 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v2 + 120), 16 * v3, 8);
    v9 = *(_QWORD *)(v2 + 88);
    if ( v9 )
      j_j___libc_free_0(v9);
    sub_C7D6A0(*(_QWORD *)(v2 + 64), 8LL * *(unsigned int *)(v2 + 80), 4);
    j_j___libc_free_0(v2);
  }
  *a1 = &unk_49DAF80;
  return sub_BB9100((__int64)a1);
}
