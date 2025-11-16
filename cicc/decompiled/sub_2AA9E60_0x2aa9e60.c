// Function: sub_2AA9E60
// Address: 0x2aa9e60
//
__int64 __fastcall sub_2AA9E60(__int64 *a1, __int64 a2, unsigned int a3, char a4)
{
  __int64 v4; // rbx
  unsigned __int64 v5; // r14
  unsigned int v6; // eax
  unsigned __int64 v7; // r13
  unsigned int v8; // r12d
  unsigned __int64 *v9; // r13
  unsigned __int64 v10; // rbx
  unsigned __int64 v12; // [rsp+0h] [rbp-70h]
  unsigned __int64 *v13; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+18h] [rbp-58h]
  unsigned __int64 *v15; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+28h] [rbp-48h]
  unsigned __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+38h] [rbp-38h]

  v4 = a3;
  if ( !a4 )
    v4 = (unsigned int)sub_DFB730(a1[56]);
  LODWORD(v5) = 0;
  sub_BCB300((__int64)&v13, *(_QWORD *)(a1[55] + 336));
  v6 = sub_DEF800(a1[53]);
  v7 = v6;
  if ( v6 )
  {
    v12 = (unsigned int)a2;
    if ( BYTE4(a2) )
    {
      v17 = sub_2AA7E40(a1[61], a1[56]);
      v5 = HIDWORD(v17);
      if ( !BYTE4(v17) )
        goto LABEL_9;
      v16 = v14;
      v12 = (unsigned int)v17 * (unsigned __int64)(unsigned int)a2;
      if ( v14 <= 0x40 )
        goto LABEL_6;
    }
    else
    {
      v16 = v14;
      if ( v14 <= 0x40 )
      {
LABEL_6:
        v15 = v13;
        goto LABEL_7;
      }
    }
    sub_C43780((__int64)&v15, (const void **)&v13);
LABEL_7:
    sub_C46F20((__int64)&v15, v7);
    v8 = v16;
    v9 = v15;
    v16 = 0;
    v10 = v12 * v4;
    v18 = v8;
    v17 = (unsigned __int64)v15;
    if ( v8 <= 0x40 )
    {
      LOBYTE(v5) = (unsigned __int64)v15 > v10;
      goto LABEL_9;
    }
    if ( v8 - (unsigned int)sub_C444A0((__int64)&v17) <= 0x40 )
    {
      LODWORD(v5) = 0;
      if ( v10 < *v9 )
      {
        LODWORD(v5) = 1;
LABEL_22:
        j_j___libc_free_0_0((unsigned __int64)v9);
        if ( v16 > 0x40 && v15 )
          j_j___libc_free_0_0((unsigned __int64)v15);
        goto LABEL_9;
      }
    }
    else
    {
      LODWORD(v5) = 1;
    }
    if ( !v9 )
      goto LABEL_9;
    goto LABEL_22;
  }
LABEL_9:
  if ( v14 > 0x40 && v13 )
    j_j___libc_free_0_0((unsigned __int64)v13);
  return (unsigned int)v5;
}
