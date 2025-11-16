// Function: sub_2AD5840
// Address: 0x2ad5840
//
__int64 *__fastcall sub_2AD5840(__int64 *a1)
{
  __int64 v1; // rax
  __int64 *result; // rax
  __int64 v3; // rax
  unsigned __int64 *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r9
  _QWORD *v8; // r12
  __int64 v9; // rax
  unsigned __int64 v10; // rcx
  __int64 *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // r8
  __int64 v16; // r15
  __int64 v17; // rax
  int v18; // ecx
  _QWORD *v19; // r12
  __int64 v20; // r12
  __int64 v21; // rax
  __int64 v22; // [rsp+0h] [rbp-A0h]
  int v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int64 v26; // [rsp+28h] [rbp-78h] BYREF
  __int64 v27; // [rsp+30h] [rbp-70h] BYREF
  __int64 v28; // [rsp+38h] [rbp-68h] BYREF
  __int64 v29[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v30; // [rsp+60h] [rbp-40h]

  v26 = **(_QWORD **)(a1[1] + 32);
  v1 = a1[5];
  if ( *(_BYTE *)(v1 + 108) && *(_DWORD *)(v1 + 100) )
  {
    v3 = sub_2BF3F10(*a1);
    v25 = sub_2BF04D0(v3);
    v4 = (unsigned __int64 *)sub_2BF05A0(v25);
    v5 = sub_2AAFF80(*a1);
    v6 = sub_22077B0(0x98u);
    v7 = v25;
    v8 = (_QWORD *)v6;
    if ( v6 )
    {
      if ( v5 )
        v5 += 96;
      v29[0] = 0;
      v28 = v5;
      sub_2AAF4A0(v6, 15, &v28, 1, v29, v25);
      sub_9C6650(v29);
      v7 = v25;
      *v8 = &unk_4A24B48;
      v8[5] = &unk_4A24B80;
      v8[12] = &unk_4A24BB8;
    }
    v8[10] = v7;
    v9 = v8[3];
    v10 = *v4;
    v8[4] = v4;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    v8[3] = v10 | v9 & 7;
    *(_QWORD *)(v10 + 8) = v8 + 3;
    *v4 = *v4 & 7 | (unsigned __int64)(v8 + 3);
    v11 = (__int64 *)a1[7];
    v12 = v11[1];
    v13 = *v11;
    v11[1] = (__int64)v4;
    *v11 = v7;
    v14 = *a1;
    v22 = v12;
    v15 = *(_QWORD *)(*a1 + 208);
    if ( !v15 )
    {
      v21 = sub_22077B0(0x38u);
      v15 = v21;
      if ( v21 )
      {
        v24 = v21;
        sub_2BF0340(v21, 0, 0, 0);
        v15 = v24;
      }
      *(_QWORD *)(v14 + 208) = v15;
    }
    v23 = v15;
    v16 = a1[7];
    v30 = 257;
    v27 = 0;
    v28 = 0;
    v17 = sub_22077B0(0xC8u);
    v18 = (_DWORD)v8 + 96;
    v19 = (_QWORD *)v17;
    if ( v17 )
    {
      sub_2C1A5F0(v17, 53, 37, v18, v23, (unsigned int)&v28, (__int64)v29);
      if ( *(_QWORD *)v16 )
        sub_2AAFF40(*(_QWORD *)v16, v19, *(unsigned __int64 **)(v16 + 8));
      v20 = (__int64)(v19 + 12);
    }
    else
    {
      v20 = *(_QWORD *)v16;
      if ( *(_QWORD *)v16 )
      {
        v20 = 0;
        sub_2AAFF40(*(_QWORD *)v16, 0, *(unsigned __int64 **)(v16 + 8));
      }
    }
    sub_9C6650(&v28);
    sub_9C6650(&v27);
    result = sub_2AD5700((__int64)(a1 + 12), &v26);
    *result = v20;
    if ( v13 )
    {
      *v11 = v13;
      v11[1] = v22;
      return (__int64 *)v22;
    }
    else
    {
      *v11 = 0;
      v11[1] = 0;
    }
  }
  else
  {
    result = sub_2AD5700((__int64)(a1 + 12), &v26);
    *result = 0;
  }
  return result;
}
