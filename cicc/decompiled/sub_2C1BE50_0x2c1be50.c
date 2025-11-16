// Function: sub_2C1BE50
// Address: 0x2c1be50
//
__int64 __fastcall sub_2C1BE50(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r15
  _QWORD *v4; // r14
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r9
  __int64 *v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r13
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  _QWORD *v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 *v19; // r10
  __int64 result; // rax
  __int64 v21; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v22; // [rsp+28h] [rbp-98h] BYREF
  __int64 v23; // [rsp+30h] [rbp-90h] BYREF
  __int64 v24; // [rsp+38h] [rbp-88h] BYREF
  __int64 v25; // [rsp+40h] [rbp-80h] BYREF
  __int64 v26; // [rsp+48h] [rbp-78h] BYREF
  __int64 v27[2]; // [rsp+50h] [rbp-70h] BYREF
  void *v28[4]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v29; // [rsp+80h] [rbp-40h]

  v3 = **(_QWORD **)(a1 + 48);
  if ( sub_2BF04A0(v3) )
  {
    v4 = (_QWORD *)sub_BD5C60(*(_QWORD *)(a1 + 96));
    v5 = sub_2BF9BD0(*(_QWORD *)(a1 + 80));
    v27[0] = v3;
    v21 = 0;
    v6 = v5;
    v29 = 257;
    v7 = sub_BCCE00(v4, 0x20u);
    v8 = sub_ACD640(v7, 1, 0);
    v27[1] = sub_2AC42A0(v6, v8);
    v22 = 0;
    v23 = 0;
    v24 = 0;
    v3 = sub_22077B0(0xC8u);
    if ( v3 )
    {
      v25 = v24;
      if ( v24 )
      {
        sub_2AAAFA0(&v25);
        v26 = v25;
        if ( v25 )
          sub_2AAAFA0(&v26);
      }
      else
      {
        v26 = 0;
      }
      sub_2AAF4A0(v3, 4, v27, 2, &v26, v9);
      sub_9C6650(&v26);
      *(_BYTE *)(v3 + 152) = 7;
      *(_DWORD *)(v3 + 156) = 0;
      *(_QWORD *)v3 = &unk_4A23258;
      *(_QWORD *)(v3 + 40) = &unk_4A23290;
      *(_QWORD *)(v3 + 96) = &unk_4A232C8;
      sub_9C6650(&v25);
      *(_BYTE *)(v3 + 160) = 82;
      *(_QWORD *)v3 = &unk_4A23B70;
      *(_QWORD *)(v3 + 40) = &unk_4A23BB8;
      *(_QWORD *)(v3 + 96) = &unk_4A23BF0;
      sub_CA0F50((__int64 *)(v3 + 168), v28);
    }
    if ( *a2 )
    {
      v10 = (__int64 *)a2[1];
      *(_QWORD *)(v3 + 80) = *a2;
      v11 = *(_QWORD *)(v3 + 24);
      v12 = *v10;
      *(_QWORD *)(v3 + 32) = v10;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v3 + 24) = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v3 + 24;
      *v10 = *v10 & 7 | (v3 + 24);
      sub_9C6650(&v24);
      sub_9C6650(&v23);
      sub_9C6650(&v22);
    }
    else
    {
      sub_9C6650(&v24);
      sub_9C6650(&v23);
      sub_9C6650(&v22);
      if ( !v3 )
        goto LABEL_10;
    }
    v3 += 96;
LABEL_10:
    sub_9C6650(&v21);
  }
  v13 = **(_QWORD **)(a1 + 48);
  v28[0] = (void *)(a1 + 40);
  v14 = *(_QWORD **)(v13 + 16);
  v15 = (__int64)&v14[*(unsigned int *)(v13 + 24)];
  v16 = sub_2C0D780(v14, v15, (__int64 *)v28);
  if ( (_QWORD *)v15 != v16 )
  {
    if ( (_QWORD *)v15 != v16 + 1 )
    {
      memmove(v16, v16 + 1, v15 - (_QWORD)(v16 + 1));
      LODWORD(v17) = *(_DWORD *)(v13 + 24);
    }
    v17 = (unsigned int)(v17 - 1);
    *(_DWORD *)(v13 + 24) = v17;
    v19 = *(__int64 **)(a1 + 48);
  }
  *v19 = v3;
  result = *(unsigned int *)(v3 + 24);
  if ( result + 1 > (unsigned __int64)*(unsigned int *)(v3 + 28) )
  {
    sub_C8D5F0(v3 + 16, (const void *)(v3 + 32), result + 1, 8u, v17, v18);
    result = *(unsigned int *)(v3 + 24);
  }
  *(_QWORD *)(*(_QWORD *)(v3 + 16) + 8 * result) = a1 + 40;
  ++*(_DWORD *)(v3 + 24);
  return result;
}
