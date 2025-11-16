// Function: sub_2427090
// Address: 0x2427090
//
_QWORD *__fastcall sub_2427090(_QWORD *a1, _BYTE *a2)
{
  _BYTE *v3; // rbx
  _BYTE *v4; // rax
  unsigned __int8 v5; // al
  unsigned __int8 v6; // dl
  __int64 *v7; // rax
  const char *v8; // r15
  size_t v9; // rdx
  size_t v10; // r13
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdi
  char v15; // cl
  _BYTE *v16; // rdx
  unsigned __int8 v17; // al
  _QWORD *v18; // rdx
  unsigned __int8 v19; // al
  __int64 *v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // rdx
  _QWORD *v24; // rbx
  __int64 v25; // rdx
  const char *v26; // rdi
  unsigned __int8 v27; // al
  __int64 v28; // rbx
  const void *v29; // [rsp+8h] [rbp-F8h]
  _QWORD v30[4]; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v31; // [rsp+30h] [rbp-D0h]
  const char *v32; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+48h] [rbp-B8h]
  __int16 v34; // [rsp+60h] [rbp-A0h]
  _BYTE v35[32]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v36; // [rsp+90h] [rbp-70h]
  const char *v37; // [rsp+A0h] [rbp-60h] BYREF
  size_t v38; // [rsp+A8h] [rbp-58h]
  __int16 v39; // [rsp+C0h] [rbp-40h]

  v3 = a2;
  *a1 = a1 + 3;
  a1[1] = 0;
  a1[2] = 128;
  v29 = a1 + 3;
  v4 = a2;
  if ( *a2 == 16 )
    goto LABEL_4;
  v5 = *(a2 - 16);
  if ( (v5 & 2) != 0 )
  {
    v4 = (_BYTE *)**((_QWORD **)a2 - 4);
    if ( v4 )
    {
LABEL_4:
      v6 = *(v4 - 16);
      if ( (v6 & 2) != 0 )
        v7 = (__int64 *)*((_QWORD *)v4 - 4);
      else
        v7 = (__int64 *)&v4[-8 * ((v6 >> 2) & 0xF) - 16];
      v8 = (const char *)*v7;
      if ( *v7 )
      {
        v8 = (const char *)sub_B91420(*v7);
        v10 = v9;
      }
      else
      {
        v10 = 0;
      }
      v37 = v8;
      v39 = 261;
      v38 = v10;
      if ( !(unsigned int)sub_C825C0((__int64)&v37, 0) )
      {
        a1[1] = 0;
        v13 = 0;
        if ( a1[2] < v10 )
        {
          sub_C8D290((__int64)a1, v29, v10, 1u, v11, v12);
          v13 = a1[1];
        }
        if ( v10 )
        {
          memcpy((void *)(*a1 + v13), v8, v10);
          v13 = a1[1];
        }
        goto LABEL_13;
      }
LABEL_19:
      v15 = *a2;
      v16 = a2;
      v39 = 257;
      v36 = 257;
      if ( v15 == 16
        || ((v17 = *(a2 - 16), (v17 & 2) != 0)
          ? (v18 = (_QWORD *)*((_QWORD *)a2 - 4))
          : (v18 = &a2[-8 * ((v17 >> 2) & 0xF) - 16]),
            (v16 = (_BYTE *)*v18) != 0) )
      {
        v19 = *(v16 - 16);
        if ( (v19 & 2) != 0 )
          v20 = (__int64 *)*((_QWORD *)v16 - 4);
        else
          v20 = (__int64 *)&v16[-8 * ((v19 >> 2) & 0xF) - 16];
        v21 = *v20;
        if ( *v20 )
        {
          v22 = sub_B91420(v21);
          v15 = *a2;
          v21 = v22;
        }
        else
        {
          v23 = 0;
        }
        v32 = (const char *)v21;
        v34 = 261;
        v33 = v23;
        if ( v15 == 16 )
          goto LABEL_34;
        v17 = *(a2 - 16);
      }
      else
      {
        v33 = 0;
        v34 = 261;
        v32 = byte_3F871B3;
      }
      if ( (v17 & 2) != 0 )
        v24 = (_QWORD *)*((_QWORD *)a2 - 4);
      else
        v24 = &a2[-8 * ((v17 >> 2) & 0xF) - 16];
      v3 = (_BYTE *)*v24;
      if ( !v3 )
      {
        v25 = 0;
        v26 = byte_3F871B3;
LABEL_38:
        v30[0] = v26;
        v31 = 261;
        v30[1] = v25;
        sub_C81B70(a1, (__int64)v30, (__int64)&v32, (__int64)v35, (__int64)&v37);
        return a1;
      }
LABEL_34:
      v27 = *(v3 - 16);
      if ( (v27 & 2) != 0 )
        v28 = *((_QWORD *)v3 - 4);
      else
        v28 = (__int64)&v3[-8 * ((v27 >> 2) & 0xF) - 16];
      v26 = *(const char **)(v28 + 8);
      if ( v26 )
        v26 = (const char *)sub_B91420((__int64)v26);
      else
        v25 = 0;
      goto LABEL_38;
    }
  }
  else
  {
    v4 = *(_BYTE **)&a2[-8 * ((v5 >> 2) & 0xF) - 16];
    if ( v4 )
      goto LABEL_4;
  }
  v38 = 0;
  v39 = 261;
  v37 = byte_3F871B3;
  if ( (unsigned int)sub_C825C0((__int64)&v37, 0) )
    goto LABEL_19;
  v10 = 0;
  v13 = 0;
LABEL_13:
  a1[1] = v13 + v10;
  return a1;
}
