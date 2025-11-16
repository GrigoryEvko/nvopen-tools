// Function: sub_30DBE70
// Address: 0x30dbe70
//
__int64 __fastcall sub_30DBE70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 **v10; // r10
  __int64 v11; // r14
  __int64 v12; // r8
  __int64 v13; // r14
  __int64 v14; // r14
  __int64 v15; // r11
  __int64 v16; // rbx
  unsigned __int8 **v17; // rcx
  int v18; // eax
  unsigned __int8 **v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r12
  int v22; // edx
  int v23; // ebx
  __int64 *v25; // rax
  bool v26; // cc
  __int64 v27; // [rsp+0h] [rbp-90h]
  __int64 **v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int8 **v30; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v31; // [rsp+38h] [rbp-58h] BYREF
  _DWORD v32[20]; // [rsp+40h] [rbp-50h] BYREF

  v6 = sub_30D92D0(a1, a2, a3, a4, a5, a6);
  if ( !(_BYTE)v6 )
  {
    v7 = sub_BCB060(*(_QWORD *)(a2 + 8));
    v8 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
      v8 = **(_QWORD **)(v8 + 16);
    if ( v7 == sub_AE2980(*(_QWORD *)(a1 + 80), *(_DWORD *)(v8 + 8) >> 8)[1] )
    {
      sub_30D74B0((__int64)&v30, a1 + 232, *(_QWORD *)(a2 - 32));
      if ( v30 )
      {
        v29 = a2;
        v25 = sub_30DA4E0(a1 + 232, &v29);
        v26 = *((_DWORD *)v25 + 4) <= 0x40u;
        *v25 = (__int64)v30;
        if ( v26 && v32[0] <= 0x40u )
        {
          v25[1] = v31;
          *((_DWORD *)v25 + 4) = v32[0];
        }
        else
        {
          sub_C43990((__int64)(v25 + 1), (__int64)&v31);
        }
      }
      if ( v32[0] > 0x40u && v31 )
        j_j___libc_free_0_0(v31);
    }
    v9 = sub_30D1740(a1, *(_QWORD *)(a2 - 32));
    if ( v9 )
    {
      v30 = (unsigned __int8 **)a2;
      *sub_30DA630(a1 + 168, (__int64 *)&v30) = v9;
    }
    v10 = *(__int64 ***)(a1 + 8);
    v11 = 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    {
      v12 = *(_QWORD *)(a2 - 8);
      v13 = v12 + v11;
    }
    else
    {
      v12 = a2 - v11;
      v13 = a2;
    }
    v14 = v13 - v12;
    v30 = (unsigned __int8 **)v32;
    v15 = v14 >> 5;
    v31 = 0x400000000LL;
    v16 = v14 >> 5;
    if ( (unsigned __int64)v14 > 0x80 )
    {
      v27 = v12;
      v28 = v10;
      sub_C8D5F0((__int64)&v30, v32, v14 >> 5, 8u, v12, (__int64)v32);
      v19 = v30;
      v18 = v31;
      v15 = v14 >> 5;
      v10 = v28;
      v12 = v27;
      v17 = &v30[(unsigned int)v31];
    }
    else
    {
      v17 = (unsigned __int8 **)v32;
      v18 = 0;
      v19 = (unsigned __int8 **)v32;
    }
    if ( v14 > 0 )
    {
      v20 = 0;
      do
      {
        v17[v20 / 8] = *(unsigned __int8 **)(v12 + 4 * v20);
        v20 += 8LL;
        --v16;
      }
      while ( v16 );
      v19 = v30;
      v18 = v31;
    }
    LODWORD(v31) = v18 + v15;
    v21 = sub_DFCEF0(v10, (unsigned __int8 *)a2, v19, (unsigned int)(v18 + v15), 3);
    v23 = v22;
    if ( v30 != (unsigned __int8 **)v32 )
      _libc_free((unsigned __int64)v30);
    if ( !v23 )
      LOBYTE(v6) = v21 == 0;
  }
  return v6;
}
