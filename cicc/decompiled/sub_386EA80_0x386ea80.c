// Function: sub_386EA80
// Address: 0x386ea80
//
void __fastcall sub_386EA80(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  char v6; // dl
  __int64 *v7; // rax
  __int64 *v8; // rsi
  unsigned int v9; // edi
  __int64 *v10; // rcx
  unsigned int v11; // edx
  __int64 v12; // rax
  bool v13; // cc
  unsigned __int64 v14; // [rsp+0h] [rbp-40h]
  unsigned __int64 v15; // [rsp+8h] [rbp-38h] BYREF
  unsigned int v16; // [rsp+10h] [rbp-30h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( v2 )
  {
    v4 = a1 + 232;
    v5 = a1 + 24;
    while ( 1 )
    {
      v7 = *(__int64 **)(a1 + 240);
      if ( *(__int64 **)(a1 + 248) != v7 )
        goto LABEL_3;
      v8 = &v7[*(unsigned int *)(a1 + 260)];
      v9 = *(_DWORD *)(a1 + 260);
      if ( v7 != v8 )
      {
        v10 = 0;
        while ( *v7 != v2 )
        {
          if ( *v7 == -2 )
            v10 = v7;
          if ( v8 == ++v7 )
          {
            if ( !v10 )
              goto LABEL_24;
            *v10 = v2;
            --*(_DWORD *)(a1 + 264);
            ++*(_QWORD *)(a1 + 232);
            goto LABEL_14;
          }
        }
        goto LABEL_4;
      }
LABEL_24:
      if ( v9 < *(_DWORD *)(a1 + 256) )
      {
        *(_DWORD *)(a1 + 260) = v9 + 1;
        *v8 = v2;
        ++*(_QWORD *)(a1 + 232);
LABEL_14:
        v14 = (4LL * *(unsigned __int8 *)(a1 + 344)) | v2 & 0xFFFFFFFFFFFFFFFBLL;
        v16 = *(_DWORD *)(a1 + 360);
        if ( v16 > 0x40 )
        {
          sub_16A4FD0((__int64)&v15, (const void **)(a1 + 352));
          v11 = *(_DWORD *)(a1 + 32);
          if ( v11 < *(_DWORD *)(a1 + 36) )
            goto LABEL_16;
        }
        else
        {
          v11 = *(_DWORD *)(a1 + 32);
          v15 = *(_QWORD *)(a1 + 352);
          if ( v11 < *(_DWORD *)(a1 + 36) )
            goto LABEL_16;
        }
        sub_386E900(v5, 0);
        v11 = *(_DWORD *)(a1 + 32);
LABEL_16:
        v12 = *(_QWORD *)(a1 + 24) + 24LL * v11;
        if ( !v12 )
        {
          v13 = v16 <= 0x40;
          *(_DWORD *)(a1 + 32) = v11 + 1;
          if ( !v13 )
          {
            if ( v15 )
              j_j___libc_free_0_0(v15);
          }
          goto LABEL_4;
        }
        *(_QWORD *)v12 = v14;
        *(_DWORD *)(v12 + 16) = v16;
        *(_QWORD *)(v12 + 8) = v15;
        ++*(_DWORD *)(a1 + 32);
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return;
      }
      else
      {
LABEL_3:
        sub_16CCBA0(v4, v2);
        if ( v6 )
          goto LABEL_14;
LABEL_4:
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return;
      }
    }
  }
}
