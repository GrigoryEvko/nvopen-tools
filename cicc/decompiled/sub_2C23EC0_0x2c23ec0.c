// Function: sub_2C23EC0
// Address: 0x2c23ec0
//
bool __fastcall sub_2C23EC0(__int64 a1, __int64 a2)
{
  bool result; // al
  __int64 v4; // rdi
  __int64 v5; // rdx
  _BYTE *v6; // rax

  result = !sub_2BF04A0(a2)
        && (v4 = *(_QWORD *)(a2 + 40)) != 0
        && (*(_BYTE *)v4 == 17
         || (v5 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v4 + 8) + 8LL) - 17, (unsigned int)v5 <= 1)
         && *(_BYTE *)v4 <= 0x15u
         && (v6 = sub_AD7630(v4, 0, v5), (v4 = (__int64)v6) != 0)
         && *v6 == 17)
        && *(_DWORD *)(v4 + 32) == 1
        && sub_1112D90(v4 + 24, a1);
  return result;
}
