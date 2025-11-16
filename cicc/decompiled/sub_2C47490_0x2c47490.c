// Function: sub_2C47490
// Address: 0x2c47490
//
__int64 __fastcall sub_2C47490(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rax
  char v6; // dl
  __int64 *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+8h] [rbp-38h]

  v2 = sub_2BF04A0(a1);
  if ( v2 )
  {
    LODWORD(v3) = 1;
    if ( *(_BYTE *)(v2 + 8) == 30 )
      return (unsigned int)v3;
  }
  v4 = sub_2BF04A0(a1);
  if ( v4 )
  {
    if ( *(_BYTE *)(v4 + 8) == 4 && *(_BYTE *)(v4 + 160) == 73 )
    {
      v13 = *(__int64 **)(v4 + 48);
      v14 = *v13;
      if ( *v13 )
      {
        v15 = v13[1];
        if ( v15 )
        {
          if ( *(_QWORD *)(a2 + 200) == v15 )
          {
            v23 = 64;
            v22 = 1;
            LODWORD(v3) = sub_2C47260((__int64)&v22, v14);
            if ( (_BYTE)v3 || (v16 = sub_2BF04A0(v14)) != 0 && *(_BYTE *)(v16 + 8) == 15 )
            {
              LODWORD(v3) = 1;
            }
            else
            {
              v17 = sub_2BF04A0(v14);
              if ( v17 && *(_BYTE *)(v17 + 8) == 33 )
              {
                LOBYTE(v18) = sub_2C1B260(v14 - 96);
                LODWORD(v3) = v18;
              }
            }
            if ( v23 > 0x40 && v22 )
              j_j___libc_free_0_0(v22);
            return (unsigned int)v3;
          }
LABEL_9:
          LODWORD(v3) = 0;
          return (unsigned int)v3;
        }
      }
    }
  }
  v5 = sub_2BF04A0(a1);
  if ( !v5 )
    goto LABEL_9;
  v6 = *(_BYTE *)(v5 + 8);
  if ( v6 != 23 )
  {
    if ( v6 == 9 )
    {
      if ( **(_BYTE **)(v5 + 136) != 82 )
        goto LABEL_9;
      goto LABEL_13;
    }
    if ( v6 != 16 )
    {
      if ( v6 != 4 || *(_BYTE *)(v5 + 160) != 53 )
        goto LABEL_9;
      goto LABEL_13;
    }
  }
  if ( *(_DWORD *)(v5 + 160) != 53 )
    goto LABEL_9;
LABEL_13:
  v8 = *(__int64 **)(v5 + 48);
  v9 = *v8;
  if ( !*v8 )
    goto LABEL_9;
  v3 = v8[1];
  if ( !v3 )
    goto LABEL_9;
  v10 = sub_2BF04A0(*v8);
  if ( !v10 || *(_BYTE *)(v10 + 8) != 15 )
  {
    v11 = sub_2BF04A0(v9);
    if ( !v11 || *(_BYTE *)(v11 + 8) != 33 || !sub_2C1B260(v9 - 96) )
      goto LABEL_9;
  }
  v12 = *(_QWORD *)(a2 + 208);
  if ( !v12 )
  {
    v19 = sub_22077B0(0x38u);
    v12 = v19;
    if ( v19 )
      sub_2BF0340(v19, 0, 0, 0, v20, v21);
    *(_QWORD *)(a2 + 208) = v12;
  }
  LOBYTE(v3) = v3 == v12;
  return (unsigned int)v3;
}
