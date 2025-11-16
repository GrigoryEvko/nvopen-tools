// Function: sub_22519B0
// Address: 0x22519b0
//
void __fastcall sub_22519B0(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v4; // rcx
  const wchar_t *v5; // rdi
  const wchar_t *v6; // rbp
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rsi
  __int64 v9; // rax
  const wchar_t *v10; // rdi
  wchar_t *v11; // r12
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  unsigned __int64 v14[4]; // [rsp+8h] [rbp-20h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 8);
  v14[0] = a2;
  if ( a2 < v4 )
  {
    v14[0] = v4;
    v2 = v4;
  }
  v5 = *(const wchar_t **)a1;
  v6 = (const wchar_t *)(a1 + 16);
  if ( a1 + 16 == *(_QWORD *)a1 )
    v7 = 3;
  else
    v7 = *(_QWORD *)(a1 + 16);
  if ( v7 != v2 )
  {
    v8 = 3;
    if ( v7 <= 3 )
      v8 = v7;
    if ( v2 <= v8 )
    {
      if ( v6 != v5 )
      {
        if ( v4 )
        {
          if ( v4 != -1 )
          {
            wmemcpy((wchar_t *)(a1 + 16), *(const wchar_t **)a1, v4 + 1);
            v5 = *(const wchar_t **)a1;
          }
        }
        else
        {
          *(_DWORD *)(a1 + 16) = *v5;
        }
        j___libc_free_0((unsigned __int64)v5);
        *(_QWORD *)a1 = v6;
      }
    }
    else
    {
      v9 = sub_22517A0(a1, v14, v7);
      v10 = *(const wchar_t **)a1;
      v11 = (wchar_t *)v9;
      v12 = *(_QWORD *)(a1 + 8);
      if ( v12 )
      {
        if ( v12 != -1 )
        {
          wmemcpy(v11, *(const wchar_t **)a1, v12 + 1);
          v10 = *(const wchar_t **)a1;
        }
      }
      else
      {
        *v11 = *v10;
      }
      if ( v6 != v10 )
        j___libc_free_0((unsigned __int64)v10);
      v13 = v14[0];
      *(_QWORD *)a1 = v11;
      *(_QWORD *)(a1 + 16) = v13;
    }
  }
}
