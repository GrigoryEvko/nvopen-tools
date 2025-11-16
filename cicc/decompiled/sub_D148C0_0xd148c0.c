// Function: sub_D148C0
// Address: 0xd148c0
//
_BYTE *__fastcall sub_D148C0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4)
{
  _BYTE *result; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // r12
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // [rsp-78h] [rbp-78h]
  __int64 v17; // [rsp-60h] [rbp-60h]
  __int64 v18; // [rsp-60h] [rbp-60h]
  __int64 v19; // [rsp-58h] [rbp-58h] BYREF
  unsigned int v20; // [rsp-50h] [rbp-50h]
  __int64 v21; // [rsp-48h] [rbp-48h] BYREF
  unsigned int v22; // [rsp-40h] [rbp-40h]

  result = *(_BYTE **)a1;
  if ( !**(_BYTE **)a1 )
  {
    *result = 1;
    v8 = sub_B43CC0(**(_QWORD **)(a1 + 8));
    v20 = a2;
    v9 = v8;
    if ( a2 > 0x40 )
    {
      sub_C43690((__int64)&v19, 0, 0);
      v22 = a2;
      sub_C43690((__int64)&v21, 0, 0);
    }
    else
    {
      v19 = 0;
      v22 = a2;
      v21 = 0;
    }
    v10 = *(_QWORD *)(a1 + 16);
    if ( *(_DWORD *)(v10 + 8) > 0x40u && *(_QWORD *)v10 )
    {
      v17 = *(_QWORD *)(a1 + 16);
      j_j___libc_free_0_0(*(_QWORD *)v10);
      v10 = v17;
    }
    *(_QWORD *)v10 = v19;
    *(_DWORD *)(v10 + 8) = v20;
    v20 = 0;
    if ( *(_DWORD *)(v10 + 24) > 0x40u && (v11 = *(_QWORD *)(v10 + 16)) != 0 )
    {
      v18 = v10;
      j_j___libc_free_0_0(v11);
      v12 = v20;
      *(_QWORD *)(v18 + 16) = v21;
      *(_DWORD *)(v18 + 24) = v22;
      if ( v12 > 0x40 && v19 )
        j_j___libc_free_0_0(v19);
    }
    else
    {
      *(_QWORD *)(v10 + 16) = v21;
      *(_DWORD *)(v10 + 24) = v22;
    }
    result = (_BYTE *)sub_9AC1B0(
                        a3,
                        *(unsigned __int64 **)(a1 + 16),
                        v9,
                        0,
                        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
                        **(_QWORD **)(a1 + 8),
                        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL),
                        1);
    if ( a4 )
    {
      v20 = a2;
      if ( a2 > 0x40 )
      {
        sub_C43690((__int64)&v19, 0, 0);
        v22 = a2;
        sub_C43690((__int64)&v21, 0, 0);
      }
      else
      {
        v19 = 0;
        v22 = a2;
        v21 = 0;
      }
      v13 = *(_QWORD *)(a1 + 32);
      if ( *(_DWORD *)(v13 + 8) > 0x40u && *(_QWORD *)v13 )
        j_j___libc_free_0_0(*(_QWORD *)v13);
      *(_QWORD *)v13 = v19;
      *(_DWORD *)(v13 + 8) = v20;
      v20 = 0;
      if ( *(_DWORD *)(v13 + 24) > 0x40u && (v14 = *(_QWORD *)(v13 + 16)) != 0 )
      {
        j_j___libc_free_0_0(v14);
        v15 = v20;
        *(_QWORD *)(v13 + 16) = v21;
        *(_DWORD *)(v13 + 24) = v22;
        if ( v15 > 0x40 )
        {
          if ( v19 )
            j_j___libc_free_0_0(v19);
        }
      }
      else
      {
        *(_QWORD *)(v13 + 16) = v21;
        *(_DWORD *)(v13 + 24) = v22;
      }
      sub_9AC1B0(
        a4,
        *(unsigned __int64 **)(a1 + 32),
        v9,
        0,
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL),
        **(_QWORD **)(a1 + 8),
        *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL),
        1);
      return (_BYTE *)v16;
    }
  }
  return result;
}
