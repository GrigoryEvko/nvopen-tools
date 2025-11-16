// Function: sub_81B120
// Address: 0x81b120
//
unsigned __int64 __fastcall sub_81B120(_DWORD *a1, int a2)
{
  bool v2; // r13
  int v3; // r15d
  char *v4; // r14
  char v5; // al
  unsigned __int64 result; // rax
  __int64 v7; // rcx
  size_t v8; // rdx
  char *v9; // rdx
  unsigned __int64 v10; // r13
  __int64 *v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v14; // [rsp+8h] [rbp-38h] BYREF

  v2 = 0;
  v3 = unk_4F0647C;
  v4 = (char *)qword_4F06410;
  if ( qword_4F064B0 )
  {
    v5 = *((_BYTE *)qword_4F064B0 + 88);
    *((_BYTE *)qword_4F064B0 + 88) = v5 | 0x20;
    v2 = (v5 & 0x20) != 0;
  }
  unk_4F063D8 = 0;
  sub_7BC390();
  if ( qword_4F064B0 )
    *((_BYTE *)qword_4F064B0 + 88) = qword_4F064B0[11] & 0xDF | (32 * v2);
  result = (unsigned int)(char)*qword_4F06460;
  if ( *qword_4F06460 != 40
    && (!a2
     || dword_4F055C0[(int)result + 128]
     || !(unsigned int)sub_7B3CF0(qword_4F06460, 0, 1)
     || (result = (unsigned __int64)qword_4F06460, *qword_4F06460 == 76)
     && ((result = (unsigned __int8)qword_4F06460[1], (_BYTE)result == 39) || (_BYTE)result == 34)) )
  {
    *a1 = 0;
    v7 = qword_4D04A00;
    qword_4F06420 = 0;
    v8 = *(_QWORD *)(qword_4D04A00 + 16);
    unk_4F06400 = v8;
    if ( unk_4F0647C != v3 || unk_4F06478 )
    {
      v4 = (char *)qword_4F195A0;
      if ( qword_4F195A8 - (__int64)qword_4F195A0 < v8 + 4 )
      {
        sub_81AC10(v8 + 4);
        v8 = unk_4F06400;
        v7 = qword_4D04A00;
        v4 = (char *)qword_4F195A0;
      }
      memcpy(v4, *(const void **)(v7 + 8), v8);
      v9 = &v4[unk_4F06400 + 4];
      *(_DWORD *)&v4[unk_4F06400] = 50332160;
      qword_4F195A0 = v9;
      sub_7AEE00(0, 0, (__int64)v4, (__int64)&v4[unk_4F06400 + 2]);
      v8 = unk_4F06400;
      qword_4F06410 = v4;
    }
    else
    {
      if ( *v4 == 10 )
      {
        v13 = (__int64 *)sub_7AF1D0((unsigned __int64)v4);
        v10 = v13[10];
        sub_7AEF90((__int64)v13);
        sub_7AEF30((__int64)&v13);
        if ( unk_4F06428 != v10 )
        {
          v11 = (__int64 *)qword_4F06440;
          v13 = (__int64 *)qword_4F06440;
          while ( v11 )
          {
            v14 = v11;
            v12 = *v11;
            v13 = (__int64 *)*v11;
            if ( v11[10] > v10 )
            {
              sub_7AEF90((__int64)v11);
              sub_7AEF30((__int64)&v14);
              v11 = v13;
            }
            else
            {
              v11 = (__int64 *)v12;
            }
          }
        }
        v8 = unk_4F06400;
      }
      qword_4F06410 = v4;
    }
    qword_4F06408 = &v4[v8 - 1];
    return (unsigned __int64)&qword_4F06408;
  }
  else
  {
    *a1 = 1;
  }
  return result;
}
