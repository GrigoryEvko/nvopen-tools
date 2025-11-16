// Function: sub_2A6A640
// Address: 0x2a6a640
//
void __fastcall sub_2A6A640(__int64 a1, unsigned __int8 *a2)
{
  unsigned __int8 **v2; // rdx
  _QWORD *v3; // rbx
  unsigned __int8 v4; // al
  __int64 *v5; // r14
  __int64 v6; // rax
  unsigned __int8 *v7; // rax
  __int64 v8; // r9
  unsigned __int8 *v9; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int8 v10; // [rsp+10h] [rbp-60h] BYREF
  char v11; // [rsp+11h] [rbp-5Fh]
  __int64 v12; // [rsp+18h] [rbp-58h] BYREF
  unsigned int v13; // [rsp+20h] [rbp-50h]
  __int64 v14; // [rsp+28h] [rbp-48h] BYREF
  unsigned int v15; // [rsp+30h] [rbp-40h]

  if ( (a2[7] & 0x40) != 0 )
    v2 = (unsigned __int8 **)*((_QWORD *)a2 - 1);
  else
    v2 = (unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  v3 = sub_2A68BC0(a1, *v2);
  v4 = *(_BYTE *)v3;
  v11 = 0;
  v10 = v4;
  if ( v4 > 3u )
  {
    if ( (unsigned __int8)(v4 - 4) > 1u )
      goto LABEL_9;
    v13 = *((_DWORD *)v3 + 4);
    if ( v13 > 0x40 )
    {
      sub_C43780((__int64)&v12, (const void **)v3 + 1);
      v15 = *((_DWORD *)v3 + 8);
      if ( v15 <= 0x40 )
        goto LABEL_7;
    }
    else
    {
      v12 = v3[1];
      v15 = *((_DWORD *)v3 + 8);
      if ( v15 <= 0x40 )
      {
LABEL_7:
        v14 = v3[3];
LABEL_8:
        v11 = *((_BYTE *)v3 + 1);
        goto LABEL_9;
      }
    }
    sub_C43780((__int64)&v14, (const void **)v3 + 3);
    goto LABEL_8;
  }
  if ( v4 > 1u )
    v12 = v3[1];
LABEL_9:
  v9 = a2;
  v5 = sub_2A686D0(a1 + 136, (__int64 *)&v9);
  if ( *(_BYTE *)v5 == 6 )
  {
    sub_2A6A450(a1, (__int64)a2);
  }
  else if ( v10 > 1u )
  {
    if ( !(unsigned __int8)sub_2A62D90((__int64)&v10)
      || (v6 = sub_2A637C0(a1, (__int64)&v10, *((_QWORD *)a2 + 1)),
          (v7 = (unsigned __int8 *)sub_96E680((unsigned int)*a2 - 29, v6)) == 0) )
    {
      sub_2A6A450(a1, (__int64)a2);
      sub_22C0090(&v10);
      return;
    }
    sub_2A63320(a1, (__int64)v5, (__int64)a2, v7, 0, v8);
  }
  sub_22C0090(&v10);
}
