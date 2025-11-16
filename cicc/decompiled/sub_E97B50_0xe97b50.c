// Function: sub_E97B50
// Address: 0xe97b50
//
void (*__fastcall sub_E97B50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6))()
{
  __int64 v8; // rbx
  unsigned int v9; // ebx
  _BYTE *v10; // rsi
  _BYTE *v11; // rax
  _BYTE *v12; // rdi
  void (*result)(); // rax
  __int64 (__fastcall *v14)(_QWORD *, __int64, _QWORD); // r14
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v18; // [rsp+8h] [rbp-68h]
  _BYTE *v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  _BYTE v22[72]; // [rsp+28h] [rbp-48h] BYREF

  v8 = *(unsigned int *)(a2 + 8);
  if ( (unsigned __int64)(v8 + 63) >> 6 == 1 )
  {
    v14 = *(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 536LL);
    if ( (unsigned int)v8 <= 0x40 )
    {
      v16 = *(_QWORD *)a2;
    }
    else
    {
      v15 = sub_C444A0(a2);
      v16 = -1;
      if ( (unsigned int)(v8 - v15) <= 0x40 )
        v16 = **(_QWORD **)a2;
    }
    return (void (*)())v14(a1, v16, (unsigned int)v8 >> 3);
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1[1] + 152LL) + 16LL) )
    {
      v18 = *(_DWORD *)(a2 + 8);
      if ( (unsigned int)v8 > 0x40 )
      {
        sub_C43780((__int64)&v17, (const void **)a2);
        LODWORD(v8) = *(_DWORD *)(a2 + 8);
      }
      else
      {
        v17 = *(_QWORD *)a2;
      }
    }
    else
    {
      sub_C496B0((__int64)&v17, a2);
      LODWORD(v8) = *(_DWORD *)(a2 + 8);
    }
    v9 = (unsigned int)v8 >> 3;
    v20 = 0;
    v19 = v22;
    v10 = v22;
    v21 = 10;
    if ( v9 )
    {
      v11 = v22;
      if ( v9 > 0xA )
      {
        sub_C8D290((__int64)&v19, v22, v9, 1u, a5, a6);
        v10 = v19;
        v11 = &v19[v20];
      }
      if ( v11 != &v10[v9] )
      {
        do
        {
          if ( v11 )
            *v11 = 0;
          ++v11;
        }
        while ( &v10[v9] != v11 );
        v10 = v19;
      }
      v20 = v9;
    }
    sub_C4E1E0((const void **)&v17, v10, v9);
    v12 = v19;
    result = *(void (**)())(*a1 + 512LL);
    if ( result != nullsub_360 )
    {
      v10 = v19;
      result = (void (*)())((__int64 (__fastcall *)(_QWORD *, _BYTE *, __int64))result)(a1, v19, v20);
      v12 = v19;
    }
    if ( v12 != v22 )
      result = (void (*)())_libc_free(v12, v10);
    if ( v18 > 0x40 )
    {
      if ( v17 )
        return (void (*)())j_j___libc_free_0_0(v17);
    }
  }
  return result;
}
