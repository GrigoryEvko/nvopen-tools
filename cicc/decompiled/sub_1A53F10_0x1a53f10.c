// Function: sub_1A53F10
// Address: 0x1a53f10
//
void __fastcall sub_1A53F10(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rbx
  _QWORD *v5; // rax
  int v6; // r9d
  unsigned __int64 v7; // r13
  void *v8; // rdi
  unsigned int v9; // r14d
  size_t v10; // rdx

  v2 = *a2;
  *a1 = *a2;
  if ( (v2 & 4) != 0 )
  {
    v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (v2 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v5 = (_QWORD *)sub_22077B0(48);
      v7 = (unsigned __int64)v5;
      if ( v5 )
      {
        v8 = v5 + 2;
        *v5 = v5 + 2;
        v5[1] = 0x400000000LL;
        v9 = *(_DWORD *)(v3 + 8);
        if ( v9 )
        {
          if ( v5 != (_QWORD *)v3 )
          {
            v10 = 8LL * v9;
            if ( v9 <= 4
              || (sub_16CD150((__int64)v5, v8, v9, 8, v9, v6),
                  v8 = *(void **)v7,
                  (v10 = 8LL * *(unsigned int *)(v3 + 8)) != 0) )
            {
              memcpy(v8, *(const void **)v3, v10);
            }
            *(_DWORD *)(v7 + 8) = v9;
          }
        }
      }
      *a1 = v7 | 4;
    }
  }
}
