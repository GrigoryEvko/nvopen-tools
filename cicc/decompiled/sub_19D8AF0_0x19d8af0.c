// Function: sub_19D8AF0
// Address: 0x19d8af0
//
__int64 __fastcall sub_19D8AF0(__int64 a1, __int64 a2, int a3)
{
  __int64 v4; // rax
  int v6; // eax
  __int64 v7; // r13
  unsigned int v8; // r14d
  int v9; // eax
  __int64 v10; // rsi
  __int64 *v11; // r8
  int v12; // ebx
  __int64 v13; // rcx
  __int64 v14; // rdx
  const void *v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rcx
  int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 *v23; // [rsp+8h] [rbp-B8h]
  __int64 v24; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v25; // [rsp+18h] [rbp-A8h]
  const void *v26; // [rsp+20h] [rbp-A0h]
  unsigned int v27; // [rsp+28h] [rbp-98h]
  __int64 v28; // [rsp+30h] [rbp-90h] BYREF
  __int64 v29; // [rsp+38h] [rbp-88h]
  const void *v30; // [rsp+40h] [rbp-80h]
  unsigned int v31; // [rsp+48h] [rbp-78h]
  __int64 v32; // [rsp+50h] [rbp-70h]
  __int64 v33; // [rsp+58h] [rbp-68h]
  const void *v34; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-58h]
  __int64 v36; // [rsp+70h] [rbp-50h]
  __int64 v37; // [rsp+78h] [rbp-48h]
  const void *v38; // [rsp+80h] [rbp-40h] BYREF
  unsigned int v39; // [rsp+88h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 8);
  if ( v4 && !*(_QWORD *)(v4 + 8) && (v6 = *(unsigned __int16 *)(a2 + 18), BYTE1(v6) &= ~0x80u, a3 == v6) )
  {
    sub_19D6800((__int64)&v24, *(_QWORD *)(a2 - 48));
    if ( v24 && *(_QWORD *)(v24 - 24LL * (*(_DWORD *)(v24 + 20) & 0xFFFFFFF)) )
    {
      sub_19D6800((__int64)&v28, *(_QWORD *)(a2 - 24));
      v7 = v28;
      if ( v28 && (v8 = v31, *(_QWORD *)(v28 - 24LL * (*(_DWORD *)(v28 + 20) & 0xFFFFFFF))) )
      {
        v9 = sub_16431D0(**(_QWORD **)(a2 - 48));
        v10 = v24;
        v36 = v7;
        v11 = (__int64 *)(a1 + 32);
        v12 = v9;
        v13 = v25;
        v39 = v8;
        v14 = v29;
        v15 = v26;
        v32 = v24;
        v38 = v30;
        v16 = v27;
        v37 = v29;
        v31 = 0;
        v33 = v25;
        v35 = v27;
        v34 = v26;
        v27 = 0;
        *(_QWORD *)a1 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_BYTE *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 32) = v10;
        *(_QWORD *)(a1 + 40) = v13;
        *(_DWORD *)(a1 + 56) = v16;
        if ( v16 > 0x40 )
        {
          sub_16A4FD0(a1 + 48, &v34);
          v7 = v36;
          v14 = v37;
          v8 = v39;
          v11 = (__int64 *)(a1 + 32);
        }
        else
        {
          *(_QWORD *)(a1 + 48) = v15;
        }
        *(_QWORD *)(a1 + 64) = v7;
        *(_QWORD *)(a1 + 72) = v14;
        *(_DWORD *)(a1 + 88) = v8;
        if ( v8 > 0x40 )
        {
          v23 = v11;
          sub_16A4FD0(a1 + 80, &v38);
          v11 = v23;
        }
        else
        {
          *(_QWORD *)(a1 + 80) = v38;
        }
        *(_DWORD *)(a1 + 96) = v12;
        if ( sub_19D6260((__int64 *)(a1 + 64), v11) )
        {
          v17 = *(_QWORD *)(a1 + 64);
          v18 = *(_QWORD *)(a1 + 72);
          v19 = *(_DWORD *)(a1 + 88);
          *(_QWORD *)(a1 + 64) = *(_QWORD *)(a1 + 32);
          v20 = *(_QWORD *)(a1 + 40);
          v21 = *(_QWORD *)(a1 + 80);
          *(_QWORD *)(a1 + 32) = v17;
          *(_QWORD *)(a1 + 72) = v20;
          v22 = *(_QWORD *)(a1 + 48);
          *(_QWORD *)(a1 + 40) = v18;
          *(_QWORD *)(a1 + 80) = v22;
          LODWORD(v22) = *(_DWORD *)(a1 + 56);
          *(_QWORD *)(a1 + 48) = v21;
          *(_DWORD *)(a1 + 88) = v22;
          *(_DWORD *)(a1 + 56) = v19;
        }
        if ( v35 > 0x40 && v34 )
          j_j___libc_free_0_0(v34);
        if ( v39 > 0x40 && v38 )
          j_j___libc_free_0_0(v38);
      }
      else
      {
        *(_QWORD *)a1 = 0;
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_BYTE *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 32) = 0;
        *(_QWORD *)(a1 + 40) = 0;
        *(_DWORD *)(a1 + 56) = 1;
        *(_QWORD *)(a1 + 48) = 0;
        *(_QWORD *)(a1 + 64) = 0;
        *(_QWORD *)(a1 + 72) = 0;
        *(_DWORD *)(a1 + 88) = 1;
        *(_QWORD *)(a1 + 80) = 0;
        *(_DWORD *)(a1 + 96) = 0;
      }
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
    }
    else
    {
      *(_QWORD *)a1 = 0;
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_BYTE *)(a1 + 24) = 0;
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_DWORD *)(a1 + 56) = 1;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 88) = 1;
      *(_QWORD *)(a1 + 80) = 0;
      *(_DWORD *)(a1 + 96) = 0;
    }
    if ( v27 > 0x40 && v26 )
      j_j___libc_free_0_0(v26);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_BYTE *)(a1 + 24) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_DWORD *)(a1 + 56) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 0;
    *(_QWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 88) = 1;
    *(_QWORD *)(a1 + 80) = 0;
    *(_DWORD *)(a1 + 96) = 0;
  }
  return a1;
}
