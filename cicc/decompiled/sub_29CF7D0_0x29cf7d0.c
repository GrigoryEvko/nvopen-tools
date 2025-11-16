// Function: sub_29CF7D0
// Address: 0x29cf7d0
//
__int64 __fastcall sub_29CF7D0(__int64 *a1, __int64 a2, unsigned __int64 **a3, _BYTE *a4)
{
  __int64 v6; // rax
  __int64 *v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rbx
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rdx
  char v13; // dl
  __int64 v14; // rax
  unsigned __int64 v16; // [rsp+0h] [rbp-90h]
  int v17; // [rsp+8h] [rbp-88h]
  __int64 *v18; // [rsp+8h] [rbp-88h]
  char v19; // [rsp+17h] [rbp-79h]
  unsigned __int64 v20; // [rsp+18h] [rbp-78h]
  __int64 v21[3]; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v22; // [rsp+40h] [rbp-50h] BYREF
  __int64 v23; // [rsp+48h] [rbp-48h]
  char v24; // [rsp+50h] [rbp-40h]

  v6 = sub_9208B0((__int64)a4, a2);
  v7 = a1;
  v23 = v8;
  v9 = *a1;
  v22 = v6;
  v19 = v8;
  if ( v9 )
  {
    v20 = (unsigned __int64)(v6 + 7) >> 3;
    do
    {
      if ( (v9 & 4) == 0 )
        break;
      v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v10 )
        break;
      v21[0] = *(_QWORD *)v10;
      sub_AE5800((__int64)&v22, (__int64)a4, v21, a3);
      if ( !v24 )
        return 0;
      v17 = v23;
      if ( (unsigned int)v23 > 0x40 )
      {
        v16 = *(unsigned int *)(v10 + 16);
        if ( v17 - (unsigned int)sub_C444A0((__int64)&v22) > 0x40 || v16 <= *(_QWORD *)v22 )
          goto LABEL_18;
      }
      else if ( *(unsigned int *)(v10 + 16) <= v22 )
      {
        goto LABEL_18;
      }
      v11 = sub_9208B0((__int64)a4, v21[0]);
      v21[2] = v12;
      v21[1] = v11;
      if ( v19 && !(_BYTE)v12 )
      {
        v13 = v24;
LABEL_23:
        if ( v13 )
        {
LABEL_18:
          v24 = 0;
          if ( (unsigned int)v23 > 0x40 && v22 )
            j_j___libc_free_0_0(v22);
        }
        return 0;
      }
      v13 = v24;
      if ( (unsigned __int64)(v11 + 7) >> 3 < v20 )
        goto LABEL_23;
      v14 = *(_QWORD *)(v10 + 8);
      if ( (unsigned int)v23 <= 0x40 )
      {
        v7 = (__int64 *)(v14 + 8 * v22);
      }
      else
      {
        v7 = (__int64 *)(v14 + 8LL * *(_QWORD *)v22);
        if ( v24 )
        {
          v18 = (__int64 *)(v14 + 8LL * *(_QWORD *)v22);
          v24 = 0;
          j_j___libc_free_0_0(v22);
          v7 = v18;
        }
      }
      v9 = *v7;
    }
    while ( *v7 );
  }
  return sub_9714E0(*v7 & 0xFFFFFFFFFFFFFFF8LL, a2, (__int64)a3, a4);
}
